"""Multimodal embedding pipeline — CLIP image+text → FAISS index."""

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "catalog.sqlite"
DEFAULT_FAISS = ROOT / "data" / "catalog.faiss"
DEFAULT_META = ROOT / "data" / "catalog_meta.parquet"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512


# ── Data loading ─────────────────────────────────────────────────────────

def get_valid_products(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return products that have a non-empty name and at least one image."""
    query = """
        SELECT p.product_id, p.merchant, p.name, p.description,
               p.price, p.currency, p.url
        FROM products p
        JOIN product_images pi ON p.product_id = pi.product_id
        WHERE p.name IS NOT NULL AND TRIM(p.name) != ''
        GROUP BY p.product_id
        HAVING COUNT(pi.image_url) > 0
        ORDER BY p.merchant, p.product_id
    """
    return pd.read_sql_query(query, conn)


def get_product_images(conn: sqlite3.Connection, product_id: str) -> list[str]:
    """Return existing local image paths for a product."""
    cur = conn.execute(
        "SELECT local_path FROM product_images WHERE product_id = ?",
        (product_id,),
    )
    paths = []
    for (local_path,) in cur:
        full = ROOT / local_path
        if full.exists():
            paths.append(str(full))
        else:
            log.debug("Image not found on disk: %s", full)
    return paths


# ── Model loading ────────────────────────────────────────────────────────

def _pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(device: torch.device):
    """Load OpenCLIP ViT-B-32 and return (model, preprocess, tokenizer)."""
    log.info("Loading CLIP model %s / %s on %s …", CLIP_MODEL, CLIP_PRETRAINED, device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    return model, preprocess, tokenizer


# ── Embedding helpers ────────────────────────────────────────────────────

@torch.no_grad()
def embed_images(
    model, preprocess, paths: list[str], device: torch.device, batch_size: int = 32,
) -> np.ndarray:
    """Encode images through CLIP. Returns (N, 512) float32 array."""
    all_feats = []
    valid_paths = []

    # Pre-load and preprocess images, skipping corrupt ones
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            valid_paths.append(p)
        except Exception as exc:
            log.warning("Skipping corrupt image %s: %s", p, exc)

    if not tensors:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    # Batch encode
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def embed_text(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    """Encode a single text string. Returns (1, 512) float32 array."""
    tokens = tokenizer([text]).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def fuse_embeddings(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Average mean-image embedding with text embedding, then L2-normalize.

    Falls back to text-only when no image embeddings are available.
    Returns (1, 512) float32 array.
    """
    if image_emb.shape[0] > 0:
        mean_img = image_emb.mean(axis=0, keepdims=True)
        fused = (mean_img + text_emb) / 2.0
    else:
        fused = text_emb.copy()

    norm = np.linalg.norm(fused, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return (fused / norm).astype(np.float32)


# ── Index building ───────────────────────────────────────────────────────

def build_index(
    db_path: Path,
    faiss_path: Path,
    meta_path: Path,
    device: torch.device,
    force: bool = False,
    batch_size: int = 32,
):
    """Main orchestrator: load data → embed → write FAISS index + metadata."""
    conn = sqlite3.connect(str(db_path))
    products = get_valid_products(conn)
    log.info("Valid products in catalog: %d", len(products))

    # Load existing metadata to support incremental builds
    existing_ids = set()
    if not force and meta_path.exists():
        old_meta = pd.read_parquet(meta_path)
        existing_ids = set(old_meta["product_id"])
        log.info("Existing embeddings: %d", len(existing_ids))

    new_products = products[~products["product_id"].isin(existing_ids)]
    if new_products.empty and not force:
        log.info("0 new products to embed — index is up to date.")
        conn.close()
        return

    if force:
        new_products = products
        log.info("Force mode: re-embedding all %d products.", len(new_products))
    else:
        log.info("New products to embed: %d", len(new_products))

    model, preprocess, tokenizer = load_clip_model(device)

    embeddings = []
    meta_rows = []

    for _, row in tqdm(new_products.iterrows(), total=len(new_products), desc="Embedding"):
        pid = row["product_id"]
        name = row["name"]
        desc = row["description"] or ""
        text_input = f"{name}. {desc[:200]}"

        # Image embeddings
        img_paths = get_product_images(conn, pid)
        img_emb = embed_images(model, preprocess, img_paths, device, batch_size)

        # Text embedding
        txt_emb = embed_text(model, tokenizer, text_input, device)

        # Fuse
        fused = fuse_embeddings(img_emb, txt_emb)
        embeddings.append(fused)
        meta_rows.append({
            "product_id": pid,
            "merchant": row["merchant"],
            "name": name,
            "price": row["price"],
            "currency": row["currency"],
            "url": row["url"],
        })

    conn.close()

    new_matrix = np.concatenate(embeddings, axis=0)

    # Merge with existing if incremental
    if not force and existing_ids and faiss_path.exists():
        old_index = faiss.read_index(str(faiss_path))
        old_matrix = faiss.rev_swig_ptr(old_index.get_xb(), old_index.ntotal * EMBED_DIM)
        old_matrix = old_matrix.reshape(old_index.ntotal, EMBED_DIM).copy()
        full_matrix = np.concatenate([old_matrix, new_matrix], axis=0)
        old_meta = pd.read_parquet(meta_path)
        new_meta_df = pd.DataFrame(meta_rows)
        full_meta = pd.concat([old_meta.drop(columns=["embed_idx"], errors="ignore"),
                               new_meta_df], ignore_index=True)
    else:
        full_matrix = new_matrix
        full_meta = pd.DataFrame(meta_rows)

    # Assign embed_idx
    full_meta["embed_idx"] = range(len(full_meta))

    # Build FAISS index (inner product on L2-normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(full_matrix)

    # Write outputs
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    full_meta.to_parquet(meta_path, index=False)

    log.info(
        "Done — index: %d vectors × %d dims → %s",
        index.ntotal, index.d, faiss_path,
    )
    log.info("Metadata: %d rows → %s", len(full_meta), meta_path)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build CLIP+FAISS embedding index")
    parser.add_argument("--force", action="store_true", help="Re-embed all products")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _pick_device(args.device)
    t0 = time.time()

    build_index(
        db_path=DEFAULT_DB,
        faiss_path=DEFAULT_FAISS,
        meta_path=DEFAULT_META,
        device=device,
        force=args.force,
        batch_size=args.batch_size,
    )

    log.info("Total time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
