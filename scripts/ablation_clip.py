"""Ablation study: CLIP embedding variants (image-only, text-only, fused).

Evaluates three embedding configurations using self-retrieval evaluation:
- Image-Only: per-product mean image embedding, fallback to text if no images
- Text-Only: product name+description text embedding only
- Fused (Image+Text): average of image and text embeddings (existing index)

Results saved to data/ablation_results.json.

Usage:
    python -m scripts.ablation_clip
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.embed import (
    DB_PATH,
    EMBED_DIM,
    ROOT,
    encode_images,
    encode_texts,
    get_valid_products,
    load_model,
    pick_device,
    prefetch_image_paths,
)
from pipelines.evaluation import evaluate_content_based
from pipelines.search import load_index

OUT_PATH = ROOT / "data" / "ablation_results.json"
IMAGE_ONLY_PATH = ROOT / "data" / "catalog_embeddings_image_only.npy"
TEXT_ONLY_PATH = ROOT / "data" / "catalog_embeddings_text_only.npy"
BATCH_SIZE = 32


def build_raw_embeddings(device):
    """Encode all images and texts, return per-product raw arrays.

    Returns:
        products: DataFrame with product_id in order matching meta
        product_img_embs: dict[int -> list[np.ndarray]] (index into products)
        text_embs: np.ndarray (N_products, 512)
        meta: pd.DataFrame from load_index() (the authoritative ordering)
    """
    model, preprocess, tokenizer = load_model(device)

    print("[1/4] Loading existing index for product ordering...", flush=True)
    _, meta = load_index()
    # meta defines the authoritative product ordering for the index

    print("[2/4] Loading valid products from database...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    products_db = get_valid_products(conn)
    print(f"      {len(products_db)} products from DB, {len(meta)} in index", flush=True)

    # Align DB products to meta ordering by product_id
    db_map = {row["product_id"]: row for _, row in products_db.iterrows()}
    # Use meta ordering — only keep products that exist in DB
    ordered_products = []
    for _, meta_row in meta.iterrows():
        pid = meta_row["product_id"]
        if pid in db_map:
            ordered_products.append(db_map[pid])
        else:
            # Product in meta but missing from DB — use meta info, no images
            ordered_products.append({
                "product_id": pid,
                "name": meta_row.get("name", ""),
                "description": "",
            })

    n_products = len(ordered_products)
    product_ids = [p["product_id"] if isinstance(p, dict) else p["product_id"]
                   for p in ordered_products]

    print("[3/4] Prefetching and encoding images...", flush=True)
    images_by_product = prefetch_image_paths(conn, product_ids)
    conn.close()

    all_image_paths = []
    image_to_product = []
    for i, pid in enumerate(product_ids):
        for p in images_by_product.get(pid, []):
            all_image_paths.append(p)
            image_to_product.append(i)

    print(f"      {len(all_image_paths)} images to encode", flush=True)
    image_embs, valid_indices = encode_images(model, preprocess, all_image_paths, device, BATCH_SIZE)

    product_img_embs: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb_row, orig_idx in zip(image_embs, valid_indices):
        product_img_embs[image_to_product[orig_idx]].append(emb_row)

    print("[4/4] Encoding texts...", flush=True)
    texts = []
    for p in ordered_products:
        if isinstance(p, dict):
            name = p.get("name", "") or ""
            desc = p.get("description", "") or ""
        else:
            name = getattr(p, "name", "") or ""
            desc = getattr(p, "description", "") or ""
        texts.append(f"{name}. {desc[:200]}")

    text_embs = encode_texts(model, tokenizer, texts, device, BATCH_SIZE)
    print(f"      {text_embs.shape[0]} texts encoded", flush=True)

    return n_products, product_img_embs, text_embs, meta


def build_image_only_matrix(n_products, product_img_embs, text_embs):
    """Build image-only embedding matrix.

    For each product:
    - If it has images: mean of image embeddings, L2-normalized
    - Fallback (no images): text embedding (already L2-normalized)
    """
    embeddings = []
    n_fallback = 0
    for i in range(n_products):
        img_list = product_img_embs.get(i, [])
        if img_list:
            mean_img = np.stack(img_list).mean(axis=0, keepdims=True)
            vec = mean_img
        else:
            # Fallback to text
            vec = text_embs[i : i + 1].copy()
            n_fallback += 1
        norm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
        embeddings.append((vec / norm).astype(np.float32))

    print(f"      Image-only: {n_products - n_fallback} image embeds, "
          f"{n_fallback} text fallbacks", flush=True)
    return np.concatenate(embeddings, axis=0)


def build_text_only_matrix(n_products, text_embs):
    """Build text-only embedding matrix.

    encode_texts() already L2-normalizes, so just return a copy.
    """
    # text_embs is already (N, 512) and L2-normalized per row
    return text_embs.copy().astype(np.float32)


def main():
    t_start = time.time()
    device = pick_device(None)
    print(f"Device: {device}", flush=True)

    # ── Build raw encodings ──────────────────────────────────────────
    n_products, product_img_embs, text_embs, meta = build_raw_embeddings(device)

    # ── Image-Only matrix ───────────────────────────────────────────
    print("\nBuilding image-only matrix...", flush=True)
    image_only_matrix = build_image_only_matrix(n_products, product_img_embs, text_embs)
    np.save(IMAGE_ONLY_PATH, image_only_matrix)
    print(f"  Saved {image_only_matrix.shape} → {IMAGE_ONLY_PATH}", flush=True)

    # ── Text-Only matrix ────────────────────────────────────────────
    print("\nBuilding text-only matrix...", flush=True)
    text_only_matrix = build_text_only_matrix(n_products, text_embs)
    np.save(TEXT_ONLY_PATH, text_only_matrix)
    print(f"  Saved {text_only_matrix.shape} → {TEXT_ONLY_PATH}", flush=True)

    # ── Load fused matrix ───────────────────────────────────────────
    print("\nLoading fused (existing) matrix...", flush=True)
    fused_matrix, meta = load_index()
    print(f"  Loaded {fused_matrix.shape}", flush=True)

    # ── Evaluate all three ──────────────────────────────────────────
    k = 10
    variants = [
        ("Image-Only", image_only_matrix),
        ("Text-Only", text_only_matrix),
        ("Fused (Image+Text)", fused_matrix),
    ]

    results_variants = []
    for name, matrix in variants:
        print(f"\nEvaluating {name}...", flush=True)
        t0 = time.time()
        metrics = evaluate_content_based(matrix, meta, k)
        elapsed = time.time() - t0
        print(f"  HR={metrics['hit_rate']:.4f}  nDCG={metrics['ndcg']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"Nov={metrics['novelty']:.3f}  ({elapsed:.1f}s)", flush=True)
        results_variants.append({
            "name": name,
            "metrics": {k_: round(float(v), 6) if isinstance(v, (float, np.floating)) else int(v)
                        for k_, v in metrics.items()},
        })

    output = {
        "k": k,
        "variants": results_variants,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
