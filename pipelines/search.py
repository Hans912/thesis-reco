"""Search utility — verify the embedding index with text or image queries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from pipelines.embed import CLIP_MODEL, CLIP_PRETRAINED, pick_device

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "catalog_embeddings.npy"
META_PATH = ROOT / "data" / "catalog_meta.parquet"


def load_index() -> tuple[np.ndarray, pd.DataFrame]:
    """Load embedding matrix and metadata."""
    matrix = np.load(EMBEDDINGS_PATH)
    meta = pd.read_parquet(META_PATH)
    return matrix, meta


def search(query_emb: np.ndarray, matrix: np.ndarray, meta: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Find top-k most similar products via cosine similarity (dot product on normalized vectors)."""
    scores = (matrix @ query_emb.T).squeeze()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = meta.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


@torch.no_grad()
def search_by_text(query: str, model, tokenizer, matrix, meta, device, top_k=5) -> pd.DataFrame:
    """Text query → embedding → cosine search."""
    tokens = tokenizer([query]).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return search(emb.cpu().numpy().astype(np.float32), matrix, meta, top_k)


@torch.no_grad()
def search_by_image(image_path: str, model, preprocess, matrix, meta, device, top_k=5) -> pd.DataFrame:
    """Image query → embedding → cosine search."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return search(emb.cpu().numpy().astype(np.float32), matrix, meta, top_k)


def main():
    parser = argparse.ArgumentParser(description="Search the CLIP embedding index")
    parser.add_argument("--query", type=str, help="Text search query")
    parser.add_argument("--image", type=str, help="Image path for visual search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    if not args.query and not args.image:
        parser.error("Provide --query or --image")

    device = pick_device(args.device)

    print("Loading model…", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    matrix, meta = load_index()
    print(f"Index: {matrix.shape[0]} products x {matrix.shape[1]} dims\n", flush=True)

    if args.query:
        results = search_by_text(args.query, model, tokenizer, matrix, meta, device, args.top_k)
    else:
        results = search_by_image(args.image, model, preprocess, matrix, meta, device, args.top_k)

    print(results[["score", "merchant", "name", "price", "currency"]].to_string(index=False))


if __name__ == "__main__":
    main()
