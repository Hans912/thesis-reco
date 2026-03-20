"""Evaluate CLIP text search with 30 curated natural-language queries.

Tests 15 fashion queries (Twinset) and 15 pet supply queries (Arcaplanet)
against the product catalog, measuring merchant hit rate and category hit rate
at k=5 and k=10.

Results saved to data/text_query_eval_results.json.

Usage:
    python -m scripts.evaluate_text_queries
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.embed import CLIP_MODEL, CLIP_PRETRAINED, pick_device
from pipelines.evaluation import extract_product_categories
from pipelines.search import load_index, search_by_text

OUT_PATH = ROOT / "data" / "text_query_eval_results.json"

# ── Query definitions ──────────────────────────────────────────────────────

QUERIES = [
    # Fashion (Twinset) — 15 queries
    {"query": "black summer dress",        "expected_merchant": "twinset", "expected_category": "vestido"},
    {"query": "floral maxi dress",         "expected_merchant": "twinset", "expected_category": "vestido"},
    {"query": "white linen trousers",      "expected_merchant": "twinset", "expected_category": "pantalon"},
    {"query": "knitted wool sweater",      "expected_merchant": "twinset", "expected_category": "jersey"},
    {"query": "leather shoulder bag",      "expected_merchant": "twinset", "expected_category": "bolso"},
    {"query": "high heeled sandals",       "expected_merchant": "twinset", "expected_category": "zapato"},
    {"query": "silk blouse",               "expected_merchant": "twinset", "expected_category": "camisa"},
    {"query": "denim jeans",               "expected_merchant": "twinset", "expected_category": "pantalon"},
    {"query": "blazer jacket",             "expected_merchant": "twinset", "expected_category": "chaqueta"},
    {"query": "mini skirt",                "expected_merchant": "twinset", "expected_category": "falda"},
    {"query": "evening dress",             "expected_merchant": "twinset", "expected_category": "vestido"},
    {"query": "casual sneakers",           "expected_merchant": "twinset", "expected_category": "zapato"},
    {"query": "cardigan pullover",         "expected_merchant": "twinset", "expected_category": "jersey"},
    {"query": "fitted bodysuit",           "expected_merchant": "twinset", "expected_category": "body"},
    {"query": "summer shorts",             "expected_merchant": "twinset", "expected_category": "pantalon"},
    # Pet supplies (Arcaplanet) — 15 queries
    {"query": "dry cat food indoor",            "expected_merchant": "arcaplanet", "expected_category": "cat_food"},
    {"query": "dog training treats snacks",     "expected_merchant": "arcaplanet", "expected_category": "dog_snack"},
    {"query": "cat litter sand",                "expected_merchant": "arcaplanet", "expected_category": "cat_litter"},
    {"query": "dog collar leash",               "expected_merchant": "arcaplanet", "expected_category": "dog_accessories"},
    {"query": "interactive cat toy",            "expected_merchant": "arcaplanet", "expected_category": "cat_toy"},
    {"query": "puppy wet food",                 "expected_merchant": "arcaplanet", "expected_category": "dog_food"},
    {"query": "cat grooming shampoo",           "expected_merchant": "arcaplanet", "expected_category": "cat_grooming"},
    {"query": "hamster cage accessories",       "expected_merchant": "arcaplanet", "expected_category": "small_animal_other"},
    {"query": "dog harness for walking",        "expected_merchant": "arcaplanet", "expected_category": "dog_accessories"},
    {"query": "aquarium fish food flakes",      "expected_merchant": "arcaplanet", "expected_category": "small_animal_food"},
    {"query": "flea treatment antiparasitic",   "expected_merchant": "arcaplanet", "expected_category": "cat_health"},
    {"query": "rabbit food pellets",            "expected_merchant": "arcaplanet", "expected_category": "small_animal_food"},
    {"query": "dog chew bone toy",              "expected_merchant": "arcaplanet", "expected_category": "dog_snack"},
    {"query": "cat scratching post",            "expected_merchant": "arcaplanet", "expected_category": "cat_accessories"},
    {"query": "bird cage food",                 "expected_merchant": "arcaplanet", "expected_category": "small_animal_other"},
]


def main():
    t_start = time.time()
    device = pick_device(None)
    print(f"Device: {device}", flush=True)

    # ── Load model ──────────────────────────────────────────────────────
    print("Loading CLIP model...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    # ── Load index and extract categories ───────────────────────────────
    print("Loading product index...", flush=True)
    matrix, meta = load_index()
    print(f"  {matrix.shape[0]} products x {matrix.shape[1]} dims", flush=True)

    categories = extract_product_categories(meta)
    # Build a product_id -> (row_index, merchant, category) lookup
    pid_to_idx = {row["product_id"]: i for i, (_, row) in enumerate(meta.iterrows())}
    pid_to_merchant = dict(zip(meta["product_id"], meta["merchant"]))
    pid_to_category = {pid: cat for pid, cat in zip(meta["product_id"], categories.values)}

    # ── Run queries ──────────────────────────────────────────────────────
    k = 10
    per_query = []

    merchant_hits_10 = []
    category_hits_10 = []
    merchant_hits_5 = []
    category_hits_5 = []

    print(f"\nRunning {len(QUERIES)} queries (k={k})...", flush=True)

    for spec in QUERIES:
        query = spec["query"]
        exp_merchant = spec["expected_merchant"]
        exp_category = spec["expected_category"]

        results_df = search_by_text(query, model, tokenizer, matrix, meta, device, top_k=k)

        top_merchants = [
            pid_to_merchant.get(row["product_id"], "")
            for _, row in results_df.iterrows()
        ]
        top_categories = [
            pid_to_category.get(row["product_id"])
            for _, row in results_df.iterrows()
        ]
        top_names = [row["name"] for _, row in results_df.head(3).iterrows()]

        # Merchant hit: any result at top-k from expected merchant?
        merchant_hit_10 = any(m == exp_merchant for m in top_merchants[:10])
        category_hit_10 = any(c == exp_category for c in top_categories[:10])
        merchant_hit_5 = any(m == exp_merchant for m in top_merchants[:5])
        category_hit_5 = any(c == exp_category for c in top_categories[:5])

        merchant_hits_10.append(merchant_hit_10)
        category_hits_10.append(category_hit_10)
        merchant_hits_5.append(merchant_hit_5)
        category_hits_5.append(category_hit_5)

        status = "OK" if category_hit_10 else ("M-OK" if merchant_hit_10 else "MISS")
        print(f"  [{status}] {query!r:45s} → cat_hit@10={category_hit_10}, "
              f"merch_hit@10={merchant_hit_10}", flush=True)

        per_query.append({
            "query": query,
            "expected_merchant": exp_merchant,
            "expected_category": exp_category,
            "merchant_hit_at_10": merchant_hit_10,
            "category_hit_at_10": category_hit_10,
            "merchant_hit_at_5": merchant_hit_5,
            "category_hit_at_5": category_hit_5,
            "top3_products": top_names,
        })

    n = len(QUERIES)
    output = {
        "n_queries": n,
        "k": k,
        "merchant_hr_at_10": round(sum(merchant_hits_10) / n, 4),
        "category_hr_at_10": round(sum(category_hits_10) / n, 4),
        "merchant_hr_at_5": round(sum(merchant_hits_5) / n, 4),
        "category_hr_at_5": round(sum(category_hits_5) / n, 4),
        "per_query": per_query,
    }

    print(f"\nSummary:")
    print(f"  Merchant HR@10:  {output['merchant_hr_at_10']:.4f}")
    print(f"  Category HR@10:  {output['category_hr_at_10']:.4f}")
    print(f"  Merchant HR@5:   {output['merchant_hr_at_5']:.4f}")
    print(f"  Category HR@5:   {output['category_hr_at_5']:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
