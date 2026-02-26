"""Run offline evaluation for all models and save results to JSON.

Usage:
    python -m scripts.run_evaluation

Produces data/evaluation_results.json with results for k=3, 5, 10.
Uses temporal train/test split (train < 2025-12-01, test >= 2025-12-01).
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
K_VALUES = [3, 5, 10]


def main():
    conn = sqlite3.connect(str(DB_PATH))

    print("Loading embeddings...", flush=True)
    matrix, meta = load_index()
    print(f"  {matrix.shape[0]} products x {matrix.shape[1]} dims")

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
        )
        all_results[str(k)] = results
        print(f"\n  Results (k={k}, {results['n_test_cases']} test cases):")
        for m in results["models"]:
            print(f"    {m['name']:25s}  HR={m['metrics']['hit_rate']:.4f}  "
                  f"nDCG={m['metrics']['ndcg']:.4f}  P@k={m['metrics']['precision']:.4f}  "
                  f"R@k={m['metrics']['recall']:.4f}")

    conn.close()

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
