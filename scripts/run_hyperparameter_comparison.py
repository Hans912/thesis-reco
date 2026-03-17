"""Offline hyperparameter comparison for ALS and LightFM WARP models.

Usage:
    python -m scripts.run_hyperparameter_comparison

Produces data/hyperparameter_results.json with results for multiple configs at k=5.
Referenced in docs/methodology.md for thesis discussion only (not shown in frontend).
"""

import json
import sqlite3
import sys
from itertools import product as itertools_product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.evaluation import (
    SPLIT_DATE,
    build_temporal_test_cases,
    build_train_interaction_matrix,
    evaluate_als,
    evaluate_lightfm,
)
from pipelines.collab_model import train_als, train_lightfm

DB_PATH = ROOT / "data" / "catalog.sqlite"
OUT_PATH = ROOT / "data" / "hyperparameter_results.json"
K = 5


def main():
    conn = sqlite3.connect(str(DB_PATH))

    print("Building temporal test cases and interaction matrix...", flush=True)
    test_cases = build_temporal_test_cases(conn, SPLIT_DATE)
    interaction_matrix, customer_ids, store_ids = build_train_interaction_matrix(conn, SPLIT_DATE)
    print(f"  {len(test_cases)} test cases, matrix: {interaction_matrix.shape}", flush=True)

    results = {"k": K, "split_date": SPLIT_DATE, "models": []}

    # ── ALS configs ──────────────────────────────────────────────────
    als_configs = list(itertools_product(
        [32, 64, 128],       # factors
        [0.01, 0.1],         # regularization
    ))
    print(f"\nTesting {len(als_configs)} ALS configurations...", flush=True)

    for factors, reg in als_configs:
        label = f"ALS (f={factors}, reg={reg})"
        print(f"  {label}...", flush=True)
        try:
            model = train_als(interaction_matrix, factors=factors, regularization=reg)
            metrics = evaluate_als(
                model, interaction_matrix, customer_ids, store_ids, test_cases, K,
            )
            results["models"].append({
                "name": label,
                "type": "ALS",
                "params": {"factors": factors, "regularization": reg},
                "metrics": {k_: round(float(v), 4) for k_, v in metrics.items()},
            })
            print(f"    HR={metrics['hit_rate']:.4f}  nDCG={metrics['ndcg']:.4f}  "
                  f"P@k={metrics['precision']:.4f}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)

    # ── LightFM WARP configs ─────────────────────────────────────────
    lfm_configs = list(itertools_product(
        [32, 64, 128],       # components
        [0.01, 0.05],        # learning_rate
    ))
    print(f"\nTesting {len(lfm_configs)} LightFM WARP configurations...", flush=True)

    for components, lr in lfm_configs:
        label = f"LightFM WARP (c={components}, lr={lr})"
        print(f"  {label}...", flush=True)
        try:
            model = train_lightfm(
                interaction_matrix,
                no_components=components,
                learning_rate=lr,
            )
            metrics = evaluate_lightfm(
                model, interaction_matrix, customer_ids, store_ids, test_cases, K,
            )
            results["models"].append({
                "name": label,
                "type": "LightFM WARP",
                "params": {"no_components": components, "learning_rate": lr},
                "metrics": {k_: round(float(v), 4) for k_, v in metrics.items()},
            })
            print(f"    HR={metrics['hit_rate']:.4f}  nDCG={metrics['ndcg']:.4f}  "
                  f"P@k={metrics['precision']:.4f}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)

    conn.close()

    # ── Find best configs ────────────────────────────────────────────
    for model_type in ["ALS", "LightFM WARP"]:
        type_results = [m for m in results["models"] if m["type"] == model_type]
        if type_results:
            best = max(type_results, key=lambda m: m["metrics"]["ndcg"])
            print(f"\nBest {model_type}: {best['name']}  "
                  f"(nDCG={best['metrics']['ndcg']:.4f})")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
