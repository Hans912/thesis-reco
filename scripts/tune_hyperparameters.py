"""Hyperparameter tuning for ALS, LightFM WARP, and LightFM Hybrid.

Uses Optuna (if installed) for Bayesian optimisation with a nested
validation split, then reports final results on the held-out test set.

Validation split design:
    Train  (tuning): transactions before 2025-10-01
    Val    (tuning): customers who visited NEW stores 2025-10-01 to 2025-11-30
    Test  (report): customers who visited NEW stores after 2025-12-01

Falls back to grid search if Optuna is not installed.

Results saved to data/hyperparameter_tuning_results.json.

Usage:
    python -m scripts.tune_hyperparameters
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from itertools import product as itertools_product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.collab_model import (
    _build_item_features,
    _build_user_features,
    train_als,
    train_lightfm,
)
from pipelines.evaluation import (
    SPLIT_DATE,
    build_temporal_test_cases,
    build_train_interaction_matrix,
    evaluate_als,
    evaluate_lightfm,
)

DB_PATH = ROOT / "data" / "catalog.sqlite"
OUT_PATH = ROOT / "data" / "hyperparameter_tuning_results.json"

VAL_SPLIT_DATE = "2025-10-01"
TEST_SPLIT_DATE = "2025-12-01"
K = 10
N_TRIALS = 40


# ── Optuna-based tuning ─────────────────────────────────────────────────────


def _als_objective(trial, interaction_matrix, customer_ids, store_ids, val_cases):
    factors = trial.suggest_int("factors", 16, 256, log=True)
    regularization = trial.suggest_float("regularization", 1e-4, 1.0, log=True)
    iterations = trial.suggest_int("iterations", 10, 50)
    try:
        model = train_als(
            interaction_matrix,
            factors=factors,
            regularization=regularization,
            iterations=iterations,
        )
        metrics = evaluate_als(model, interaction_matrix, customer_ids, store_ids, val_cases, K)
        return metrics["hit_rate"]
    except Exception as exc:
        # Return 0 so Optuna marks this trial as a failure gracefully
        print(f"    ALS trial failed: {exc}", flush=True)
        return 0.0


def _lightfm_objective(trial, interaction_matrix, customer_ids, store_ids, val_cases,
                        item_features=None, user_features=None):
    no_components = trial.suggest_int("no_components", 16, 256, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
    epochs = trial.suggest_int("epochs", 10, 100)
    try:
        model = train_lightfm(
            interaction_matrix,
            item_features=item_features,
            user_features=user_features,
            no_components=no_components,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        metrics = evaluate_lightfm(
            model, interaction_matrix, customer_ids, store_ids, val_cases, K,
            item_features=item_features, user_features=user_features,
        )
        return metrics["hit_rate"]
    except Exception as exc:
        print(f"    LightFM trial failed: {exc}", flush=True)
        return 0.0


def tune_with_optuna(conn):
    """Run Optuna Bayesian optimisation on the validation split."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[Optuna] Building validation data (val split: {VAL_SPLIT_DATE})...", flush=True)
    val_cases = build_temporal_test_cases(conn, VAL_SPLIT_DATE)
    val_matrix, val_customer_ids, val_store_ids = build_train_interaction_matrix(conn, VAL_SPLIT_DATE)
    print(f"         {len(val_cases)} validation cases, matrix: {val_matrix.shape}", flush=True)

    # Build features for hybrid (from val train data)
    print("[Optuna] Building item and user features for hybrid...", flush=True)
    item_features = _build_item_features(val_store_ids, conn)
    user_features = _build_user_features(val_customer_ids, conn, before_date=VAL_SPLIT_DATE)

    model_results = []

    # ── ALS ────────────────────────────────────────────────────────
    print(f"\n[Optuna] Tuning ALS ({N_TRIALS} trials)...", flush=True)
    t0 = time.time()
    als_study = optuna.create_study(direction="maximize")
    als_study.optimize(
        lambda trial: _als_objective(trial, val_matrix, val_customer_ids, val_store_ids, val_cases),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    als_best = als_study.best_params
    als_val_hr = round(als_study.best_value, 6)
    print(f"         Best ALS params: {als_best}  val_HR@{K}={als_val_hr:.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ── LightFM WARP ───────────────────────────────────────────────
    print(f"\n[Optuna] Tuning LightFM WARP ({N_TRIALS} trials)...", flush=True)
    t0 = time.time()
    warp_study = optuna.create_study(direction="maximize")
    warp_study.optimize(
        lambda trial: _lightfm_objective(
            trial, val_matrix, val_customer_ids, val_store_ids, val_cases,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    warp_best = warp_study.best_params
    warp_val_hr = round(warp_study.best_value, 6)
    print(f"         Best WARP params: {warp_best}  val_HR@{K}={warp_val_hr:.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ── LightFM Hybrid ─────────────────────────────────────────────
    print(f"\n[Optuna] Tuning LightFM Hybrid ({N_TRIALS} trials)...", flush=True)
    t0 = time.time()
    hybrid_study = optuna.create_study(direction="maximize")
    hybrid_study.optimize(
        lambda trial: _lightfm_objective(
            trial, val_matrix, val_customer_ids, val_store_ids, val_cases,
            item_features=item_features, user_features=user_features,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    hybrid_best = hybrid_study.best_params
    hybrid_val_hr = round(hybrid_study.best_value, 6)
    print(f"         Best Hybrid params: {hybrid_best}  val_HR@{K}={hybrid_val_hr:.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    return [
        {"name": "ALS", "best_params": als_best, "val_hr_at_10": als_val_hr},
        {"name": "LightFM WARP", "best_params": warp_best, "val_hr_at_10": warp_val_hr},
        {"name": "LightFM Hybrid", "best_params": hybrid_best, "val_hr_at_10": hybrid_val_hr},
    ]


def tune_with_grid_search(conn):
    """Fallback grid search when Optuna is not available."""
    print("[Grid] Building validation data (val split: {VAL_SPLIT_DATE})...", flush=True)
    val_cases = build_temporal_test_cases(conn, VAL_SPLIT_DATE)
    val_matrix, val_customer_ids, val_store_ids = build_train_interaction_matrix(conn, VAL_SPLIT_DATE)
    print(f"       {len(val_cases)} validation cases, matrix: {val_matrix.shape}", flush=True)

    best_als_params = None
    best_als_hr = -1.0

    als_grid = list(itertools_product(
        [32, 64, 128, 256],
        [0.001, 0.01, 0.1],
        [20, 30, 50],
    ))
    print(f"\n[Grid] ALS: {len(als_grid)} configurations...", flush=True)
    for factors, reg, iters in als_grid:
        try:
            model = train_als(val_matrix, factors=factors, regularization=reg, iterations=iters)
            metrics = evaluate_als(model, val_matrix, val_customer_ids, val_store_ids, val_cases, K)
            hr = metrics["hit_rate"]
            print(f"  ALS f={factors}, reg={reg}, iter={iters} → HR@{K}={hr:.4f}", flush=True)
            if hr > best_als_hr:
                best_als_hr = hr
                best_als_params = {"factors": factors, "regularization": reg, "iterations": iters}
        except Exception as exc:
            print(f"  ALS f={factors}, reg={reg}, iter={iters} FAILED: {exc}", flush=True)

    best_warp_params = None
    best_warp_hr = -1.0

    warp_grid = list(itertools_product(
        [32, 64, 128, 256],
        [0.01, 0.05, 0.1],
        [20, 50, 100],
    ))
    print(f"\n[Grid] LightFM WARP: {len(warp_grid)} configurations...", flush=True)
    for components, lr, epochs in warp_grid:
        try:
            model = train_lightfm(val_matrix, no_components=components,
                                  learning_rate=lr, epochs=epochs)
            metrics = evaluate_lightfm(model, val_matrix, val_customer_ids,
                                       val_store_ids, val_cases, K)
            hr = metrics["hit_rate"]
            print(f"  WARP c={components}, lr={lr}, ep={epochs} → HR@{K}={hr:.4f}", flush=True)
            if hr > best_warp_hr:
                best_warp_hr = hr
                best_warp_params = {"no_components": components,
                                    "learning_rate": lr, "epochs": epochs}
        except Exception as exc:
            print(f"  WARP c={components}, lr={lr}, ep={epochs} FAILED: {exc}", flush=True)

    # Hybrid uses same architecture as WARP but with features
    print("\n[Grid] Building item and user features for hybrid...", flush=True)
    item_features = _build_item_features(val_store_ids, conn)
    user_features = _build_user_features(val_customer_ids, conn, before_date=VAL_SPLIT_DATE)

    best_hybrid_params = None
    best_hybrid_hr = -1.0

    hybrid_grid = list(itertools_product(
        [64, 128, 256],
        [0.005, 0.01, 0.05],
        [30, 50, 100],
    ))
    print(f"\n[Grid] LightFM Hybrid: {len(hybrid_grid)} configurations...", flush=True)
    for components, lr, epochs in hybrid_grid:
        try:
            model = train_lightfm(val_matrix, item_features=item_features,
                                  user_features=user_features,
                                  no_components=components,
                                  learning_rate=lr, epochs=epochs)
            metrics = evaluate_lightfm(model, val_matrix, val_customer_ids,
                                       val_store_ids, val_cases, K,
                                       item_features=item_features,
                                       user_features=user_features)
            hr = metrics["hit_rate"]
            print(f"  Hybrid c={components}, lr={lr}, ep={epochs} → HR@{K}={hr:.4f}", flush=True)
            if hr > best_hybrid_hr:
                best_hybrid_hr = hr
                best_hybrid_params = {"no_components": components,
                                      "learning_rate": lr, "epochs": epochs}
        except Exception as exc:
            print(f"  Hybrid c={components}, lr={lr}, ep={epochs} FAILED: {exc}", flush=True)

    return [
        {"name": "ALS",
         "best_params": best_als_params or {"factors": 64, "regularization": 0.01, "iterations": 30},
         "val_hr_at_10": round(best_als_hr, 6)},
        {"name": "LightFM WARP",
         "best_params": best_warp_params or {"no_components": 64, "learning_rate": 0.05, "epochs": 30},
         "val_hr_at_10": round(best_warp_hr, 6)},
        {"name": "LightFM Hybrid",
         "best_params": best_hybrid_params or {"no_components": 128, "learning_rate": 0.01, "epochs": 50},
         "val_hr_at_10": round(best_hybrid_hr, 6)},
    ]


# ── Final evaluation on test set ────────────────────────────────────────────


def evaluate_on_test(best_results, conn):
    """Retrain each model on full train (< 2025-12-01) and evaluate on test set."""
    print(f"\nBuilding full train matrix (before {TEST_SPLIT_DATE})...", flush=True)
    test_cases = build_temporal_test_cases(conn, TEST_SPLIT_DATE)
    train_matrix, customer_ids, store_ids = build_train_interaction_matrix(conn, TEST_SPLIT_DATE)
    print(f"  {len(test_cases)} test cases, matrix: {train_matrix.shape}", flush=True)

    # Build features from full train data (for hybrid)
    print("Building item and user features for hybrid (full train)...", flush=True)
    item_features = _build_item_features(store_ids, conn)
    user_features = _build_user_features(customer_ids, conn, before_date=TEST_SPLIT_DATE)

    for entry in best_results:
        name = entry["name"]
        params = entry["best_params"]
        print(f"\nRetraining {name} with best params on full train...", flush=True)
        print(f"  Params: {params}", flush=True)
        t0 = time.time()
        try:
            if name == "ALS":
                model = train_als(train_matrix, **params)
                metrics = evaluate_als(model, train_matrix, customer_ids, store_ids,
                                       test_cases, K)
            elif name == "LightFM WARP":
                model = train_lightfm(train_matrix, **params)
                metrics = evaluate_lightfm(model, train_matrix, customer_ids, store_ids,
                                           test_cases, K)
            elif name == "LightFM Hybrid":
                model = train_lightfm(train_matrix, item_features=item_features,
                                      user_features=user_features, **params)
                metrics = evaluate_lightfm(model, train_matrix, customer_ids, store_ids,
                                           test_cases, K,
                                           item_features=item_features,
                                           user_features=user_features)
            else:
                continue

            entry["test_hr_at_10"] = round(float(metrics["hit_rate"]), 6)
            entry["test_ndcg_at_10"] = round(float(metrics["ndcg"]), 6)
            entry["test_precision_at_10"] = round(float(metrics["precision"]), 6)
            entry["test_recall_at_10"] = round(float(metrics["recall"]), 6)
            entry["test_coverage"] = round(float(metrics["coverage"]), 6)
            entry["n_test_cases"] = len(test_cases)
            print(f"  HR@{K}={metrics['hit_rate']:.4f}  nDCG@{K}={metrics['ndcg']:.4f}  "
                  f"P@{K}={metrics['precision']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            entry["test_hr_at_10"] = None
            entry["test_ndcg_at_10"] = None

    return best_results


def load_default_test_results():
    """Load default-param test results from evaluation_results.json if available."""
    eval_path = ROOT / "data" / "evaluation_results.json"
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path) as f:
            all_results = json.load(f)
        # Use k=10 results
        k10 = all_results.get("10", {})
        models = k10.get("models", [])
        defaults = {}
        for m in models:
            n = m.get("name", "")
            if "ALS" in n and "Hybrid" not in n and "LightFM" not in n:
                defaults["ALS"] = m["metrics"].get("hit_rate")
            elif "LightFM WARP" in n and "Hybrid" not in n:
                defaults["LightFM WARP"] = m["metrics"].get("hit_rate")
            elif "LightFM Hybrid" in n or ("LightFM" in n and "Hybrid" in n):
                defaults["LightFM Hybrid"] = m["metrics"].get("hit_rate")
        return defaults
    except Exception:
        return {}


def main():
    t_start = time.time()
    conn = sqlite3.connect(str(DB_PATH))

    # Detect whether Optuna is available
    use_optuna = False
    try:
        import optuna  # noqa: F401
        use_optuna = True
        print("Optuna available — using Bayesian optimisation.", flush=True)
    except ImportError:
        print("Optuna not installed — falling back to grid search.", flush=True)
        print("To install: pip install optuna", flush=True)

    # ── Run tuning ─────────────────────────────────────────────────────
    if use_optuna:
        best_results = tune_with_optuna(conn)
    else:
        best_results = tune_with_grid_search(conn)

    # ── Evaluate on test set ───────────────────────────────────────────
    best_results = evaluate_on_test(best_results, conn)

    # ── Load default baselines for comparison ──────────────────────────
    default_results = load_default_test_results()
    for entry in best_results:
        entry["default_test_hr_at_10"] = default_results.get(entry["name"])

    conn.close()

    # ── Save results ───────────────────────────────────────────────────
    output = {
        "tuning_method": "optuna_bayesian" if use_optuna else "grid_search",
        "validation_split_date": VAL_SPLIT_DATE,
        "test_split_date": TEST_SPLIT_DATE,
        "n_trials_per_model": N_TRIALS if use_optuna else None,
        "k": K,
        "elapsed_total_s": round(time.time() - t_start, 1),
        "models": best_results,
    }

    print(f"\n{'='*60}")
    print(f"Tuning method: {output['tuning_method']}")
    for entry in best_results:
        name = entry["name"]
        val_hr = entry.get("val_hr_at_10", "N/A")
        test_hr = entry.get("test_hr_at_10", "N/A")
        default_hr = entry.get("default_test_hr_at_10", "N/A")
        print(f"  {name:20s}: val_HR={val_hr}  test_HR={test_hr}  default_test_HR={default_hr}")
        print(f"               params: {entry['best_params']}")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
