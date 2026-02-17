"""Multimodal embedding pipeline — CLIP image+text → numpy index."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "catalog.sqlite"
EMBEDDINGS_PATH = ROOT / "data" / "catalog_embeddings.npy"
META_PATH = ROOT / "data" / "catalog_meta.parquet"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512


# ── Helpers ──────────────────────────────────────────────────────────────

def pick_device(requested: str | None) -> torch.device:
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    """Load OpenCLIP ViT-B-32 and return (model, preprocess, tokenizer)."""
    print(f"[1/6] Loading model…", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    print(f"      Done in {time.time() - t0:.1f}s (device={device})", flush=True)
    return model, preprocess, tokenizer


def get_valid_products(conn: sqlite3.Connection) -> pd.DataFrame:
    """Query products with non-empty names and at least one image."""
    return pd.read_sql_query("""
        SELECT p.product_id, p.merchant, p.name, p.description,
               p.price, p.currency, p.url
        FROM products p
        JOIN product_images pi ON p.product_id = pi.product_id
        WHERE p.name IS NOT NULL AND TRIM(p.name) != ''
        GROUP BY p.product_id
        HAVING COUNT(pi.image_url) > 0
        ORDER BY p.merchant, p.product_id
    """, conn)


def prefetch_image_paths(
    conn: sqlite3.Connection, product_ids: list[str],
) -> dict[str, list[str]]:
    """Fetch all image paths for given products in one query."""
    placeholders = ",".join("?" for _ in product_ids)
    rows = conn.execute(
        f"SELECT product_id, local_path FROM product_images "
        f"WHERE product_id IN ({placeholders})",
        product_ids,
    ).fetchall()

    result: dict[str, list[str]] = defaultdict(list)
    missing = 0
    for pid, local_path in rows:
        full = ROOT / local_path
        if full.exists():
            result[pid].append(str(full))
        else:
            missing += 1

    if missing:
        print(f"      Warning: {missing} images not found on disk", flush=True)
    return dict(result)


@torch.no_grad()
def encode_images(
    model, preprocess, paths: list[str], device: torch.device, batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    """Encode all images in batches. Returns (embeddings, valid_indices)."""
    tensors = []
    valid_indices = []
    for i, p in enumerate(tqdm(paths, desc="Loading images", unit="img")):
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            valid_indices.append(i)
        except Exception as exc:
            log.warning("Skipping corrupt image %s: %s", p, exc)

    if not tensors:
        return np.empty((0, EMBED_DIM), dtype=np.float32), []

    all_feats = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Encoding images", unit="batch"):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(all_feats, axis=0), valid_indices


@torch.no_grad()
def encode_texts(
    model, tokenizer, texts: list[str], device: torch.device, batch_size: int,
) -> np.ndarray:
    """Encode all texts in batches. Returns (N, 512) array."""
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", unit="batch"):
        tokens = tokenizer(texts[i : i + batch_size]).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(all_feats, axis=0)


def fuse_product_embeddings(
    product_img_embs: dict[int, list[np.ndarray]],
    text_embs: np.ndarray,
    n_products: int,
) -> np.ndarray:
    """Fuse image + text embeddings per product. Returns (N, 512) array."""
    embeddings = []
    for i in range(n_products):
        img_list = product_img_embs.get(i, [])
        txt_vec = text_embs[i : i + 1]
        if img_list:
            mean_img = np.stack(img_list).mean(axis=0, keepdims=True)
            fused = (mean_img + txt_vec) / 2.0
        else:
            fused = txt_vec.copy()
        norm = np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
        embeddings.append((fused / norm).astype(np.float32))

    return np.concatenate(embeddings, axis=0)


# ── Main pipeline ────────────────────────────────────────────────────────

def build_index(device: torch.device, force: bool = False, batch_size: int = 32):
    """Build embedding index: load data → encode → save."""

    # Step 1: Load model
    model, preprocess, tokenizer = load_model(device)

    # Step 2: Load products
    print("[2/6] Loading products from database…", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    products = get_valid_products(conn)
    print(f"      {len(products)} valid products", flush=True)

    # Incremental check
    existing_ids = set()
    if not force and META_PATH.exists():
        old_meta = pd.read_parquet(META_PATH)
        existing_ids = set(old_meta["product_id"])
        print(f"      {len(existing_ids)} already embedded", flush=True)

    new_products = products[~products["product_id"].isin(existing_ids)]
    if new_products.empty:
        print("      Index is up to date. Use --force to rebuild.", flush=True)
        conn.close()
        return

    print(f"      {len(new_products)} products to embed", flush=True)

    # Step 3: Prefetch image paths
    print("[3/6] Prefetching image paths…", flush=True)
    product_ids = new_products["product_id"].tolist()
    images_by_product = prefetch_image_paths(conn, product_ids)
    conn.close()

    all_image_paths = []
    image_to_product = []
    for i, pid in enumerate(product_ids):
        for p in images_by_product.get(pid, []):
            all_image_paths.append(p)
            image_to_product.append(i)

    print(f"      {len(all_image_paths)} images to encode", flush=True)

    # Step 4: Encode images
    print("[4/6] Encoding images…", flush=True)
    t0 = time.time()
    image_embs, valid_indices = encode_images(
        model, preprocess, all_image_paths, device, batch_size,
    )

    product_img_embs: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb_row, orig_idx in zip(image_embs, valid_indices):
        product_img_embs[image_to_product[orig_idx]].append(emb_row)

    print(f"      {image_embs.shape[0]} images encoded in {time.time() - t0:.1f}s", flush=True)

    # Step 5: Encode texts
    print("[5/6] Encoding texts…", flush=True)
    t0 = time.time()
    texts = [
        f"{row['name']}. {(row['description'] or '')[:200]}"
        for _, row in new_products.iterrows()
    ]
    text_embs = encode_texts(model, tokenizer, texts, device, batch_size)
    print(f"      {text_embs.shape[0]} texts encoded in {time.time() - t0:.1f}s", flush=True)

    # Fuse
    print("      Fusing embeddings…", flush=True)
    new_matrix = fuse_product_embeddings(product_img_embs, text_embs, len(product_ids))

    # Step 6: Save
    print("[6/6] Saving index…", flush=True)
    meta_rows = [
        {"product_id": r["product_id"], "merchant": r["merchant"], "name": r["name"],
         "price": r["price"], "currency": r["currency"], "url": r["url"]}
        for _, r in new_products.iterrows()
    ]

    # Merge with existing if incremental
    if not force and existing_ids and EMBEDDINGS_PATH.exists():
        old_matrix = np.load(EMBEDDINGS_PATH)
        full_matrix = np.concatenate([old_matrix, new_matrix], axis=0)
        old_meta = pd.read_parquet(META_PATH)
        full_meta = pd.concat([old_meta, pd.DataFrame(meta_rows)], ignore_index=True)
    else:
        full_matrix = new_matrix
        full_meta = pd.DataFrame(meta_rows)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, full_matrix)
    full_meta.to_parquet(META_PATH, index=False)

    print(f"      Embeddings: {full_matrix.shape} → {EMBEDDINGS_PATH}", flush=True)
    print(f"      Metadata:   {len(full_meta)} rows → {META_PATH}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build CLIP embedding index")
    parser.add_argument("--force", action="store_true", help="Re-embed all products")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Silence noisy loggers
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    device = pick_device(args.device)
    t0 = time.time()
    build_index(device=device, force=args.force, batch_size=args.batch_size)
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
