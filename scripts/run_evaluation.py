"""Run offline evaluation for all models and save results to JSON.

Usage:
    python -m scripts.run_evaluation

Produces data/evaluation_results.json with results for k=3, 5, 10, 15, 20.
Uses temporal train/test split (train < 2025-12-01, test >= 2025-12-01).

Demographic models (Demographic Popularity, LightFM Demo, LightFM Full
Hybrid) are included automatically when the customer_demographics table
exists in the database.  Run populate_customer_demographics first:
    python -m scripts.populate_customer_demographics
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.evaluation import run_full_evaluation
from pipelines.search import load_index

DB_PATH = ROOT / "data" / "catalog.sqlite"
OUT_PATH = ROOT / "data" / "evaluation_results.json"
K_VALUES = [3, 5, 10, 15, 20]


def main():
    conn = sqlite3.connect(str(DB_PATH))

    print("Loading embeddings...", flush=True)
    matrix, meta = load_index()
    print(f"  {matrix.shape[0]} products x {matrix.shape[1]} dims")

    # Load demographics if table exists
    demo_dict = None
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='customer_demographics'"
    ).fetchone()
    if table_exists:
        from pipelines.demographic import get_customer_demographics
        demo_dict = get_customer_demographics(conn)
        print(f"  Loaded demographics for {len(demo_dict)} customers")
        print("  Demographic models will be included in evaluation.")
    else:
        print("  customer_demographics table not found — skipping demographic models.")
        print("  Run: python -m scripts.populate_customer_demographics")

    all_results = {}
    for k in K_VALUES:
        print(f"\n{'='*60}", flush=True)
        print(f"Running evaluation for k={k}...", flush=True)
        print(f"{'='*60}", flush=True)
        results = run_full_evaluation(
            embedding_matrix=matrix,
            meta=meta,
            conn=conn,
            k=k,
            demo_dict=demo_dict,
        )
        all_results[str(k)] = results
        print(f"\n  Results (k={k}, {results['n_test_cases']} test cases):")
        for m in results["models"]:
            novelty_str = (f"  Nov={m['metrics']['novelty']:.3f}"
                           if "novelty" in m["metrics"] else "")
            print(f"    {m['name']:35s}  HR={m['metrics']['hit_rate']:.4f}  "
                  f"nDCG={m['metrics']['ndcg']:.4f}  P@k={m['metrics']['precision']:.4f}  "
                  f"R@k={m['metrics']['recall']:.4f}{novelty_str}")

    conn.close()

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
