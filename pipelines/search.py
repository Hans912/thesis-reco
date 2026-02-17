"""Search utility — verify the FAISS index with text or image queries."""

import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pipelines.embed import _pick_device, load_clip_model

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAISS = ROOT / "data" / "catalog.faiss"
DEFAULT_META = ROOT / "data" / "catalog_meta.parquet"


def load_index(faiss_path: Path, meta_path: Path):
    """Load FAISS index and metadata from disk."""
    index = faiss.read_index(str(faiss_path))
    meta = pd.read_parquet(meta_path)
    log.info("Loaded index: %d vectors × %d dims", index.ntotal, index.d)
    return index, meta


@torch.no_grad()
def search_by_text(query: str, model, tokenizer, index, meta, device, top_k=5) -> pd.DataFrame:
    """Text query → CLIP embedding → FAISS search → ranked results."""
    tokens = tokenizer([query]).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy().astype(np.float32)

    scores, idxs = index.search(emb, top_k)
    results = meta.iloc[idxs[0]].copy()
    results["score"] = scores[0]
    return results


@torch.no_grad()
def search_by_image(image_path: str, model, preprocess, index, meta, device, top_k=5) -> pd.DataFrame:
    """Image query → CLIP embedding → FAISS search → ranked results."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy().astype(np.float32)

    scores, idxs = index.search(emb, top_k)
    results = meta.iloc[idxs[0]].copy()
    results["score"] = scores[0]
    return results


def main():
    parser = argparse.ArgumentParser(description="Search the CLIP+FAISS index")
    parser.add_argument("--query", type=str, help="Text search query")
    parser.add_argument("--image", type=str, help="Image file path for visual search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.query and not args.image:
        parser.error("Provide --query or --image")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _pick_device(args.device)
    model, preprocess, tokenizer = load_clip_model(device)
    index, meta = load_index(DEFAULT_FAISS, DEFAULT_META)

    if args.query:
        results = search_by_text(args.query, model, tokenizer, index, meta, device, args.top_k)
    else:
        results = search_by_image(args.image, model, preprocess, index, meta, device, args.top_k)

    print("\n" + results[["score", "merchant", "name", "price", "currency", "url"]].to_string(index=False))


if __name__ == "__main__":
    main()
