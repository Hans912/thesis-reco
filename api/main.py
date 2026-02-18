"""FastAPI recommendation API — loads CLIP model once, serves similarity search."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

from api.models import (
    ProductDetail,
    ProductListResponse,
    ProductSummary,
    SearchResponse,
    SearchResult,
)
from pipelines.embed import DB_PATH, ROOT, pick_device
from pipelines.search import (
    load_index,
    search,
    search_by_image,
    search_by_text,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _image_urls_for_product(conn: sqlite3.Connection, product_id: str) -> List[str]:
    """Return API-relative image URLs for a product."""
    rows = conn.execute(
        "SELECT local_path FROM product_images WHERE product_id = ? ORDER BY local_path",
        (product_id,),
    ).fetchall()
    urls = []
    for (local_path,) in rows:
        full = ROOT / local_path
        if full.exists():
            urls.append("/" + local_path.removeprefix("data/"))
        # else: image missing on disk, skip
    return urls


def _first_image_url(conn: sqlite3.Connection, product_id: str) -> Optional[str]:
    """Return the first available image URL for a product, or None."""
    row = conn.execute(
        "SELECT local_path FROM product_images WHERE product_id = ? ORDER BY local_path LIMIT 1",
        (product_id,),
    ).fetchone()
    if row:
        full = ROOT / row[0]
        if full.exists():
            return "/" + row[0].removeprefix("data/")
    return None


def _row_to_summary(row: dict, image_url: Optional[str]) -> ProductSummary:
    return ProductSummary(
        product_id=row["product_id"],
        merchant=row["merchant"],
        name=row["name"],
        price=row.get("price"),
        currency=row.get("currency"),
        url=row["url"],
        image_url=image_url,
    )


def _df_to_search_results(df, conn: sqlite3.Connection) -> List[SearchResult]:
    """Convert a search results DataFrame to a list of SearchResult models."""
    results = []
    for _, row in df.iterrows():
        img = _first_image_url(conn, row["product_id"])
        results.append(SearchResult(
            product_id=row["product_id"],
            merchant=row["merchant"],
            name=row["name"],
            price=row.get("price"),
            currency=row.get("currency"),
            url=row["url"],
            image_url=img,
            score=float(row["score"]),
        ))
    return results


# ── App setup ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, index, and database connection at startup."""
    import open_clip

    from pipelines.embed import CLIP_MODEL, CLIP_PRETRAINED

    device = pick_device(None)
    print(f"Loading CLIP model on {device}…", flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    matrix, meta = load_index()
    print(f"Index loaded: {matrix.shape[0]} products x {matrix.shape[1]} dims", flush=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    app.state.model = model
    app.state.preprocess = preprocess
    app.state.tokenizer = tokenizer
    app.state.matrix = matrix
    app.state.meta = meta
    app.state.device = device
    app.state.conn = conn

    yield

    conn.close()


app = FastAPI(title="Thesis Reco API", lifespan=lifespan)

# Serve product images as static files
IMAGES_DIR = ROOT / "data" / "images"
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/products", response_model=ProductListResponse)
def list_products(
    merchant: Optional[str] = Query(None, description="Filter by merchant"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Browse the product catalog with optional merchant filter."""
    meta = app.state.meta
    if merchant:
        meta = meta[meta["merchant"] == merchant]

    total = len(meta)
    page = meta.iloc[offset : offset + limit]

    products = []
    for _, row in page.iterrows():
        img = _first_image_url(app.state.conn, row["product_id"])
        products.append(_row_to_summary(row.to_dict(), img))

    return ProductListResponse(total=total, offset=offset, limit=limit, products=products)


@app.get("/api/products/{product_id}", response_model=ProductDetail)
def get_product(product_id: str):
    """Get a single product with all images."""
    meta = app.state.meta
    match = meta[meta["product_id"] == product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    row = match.iloc[0].to_dict()
    image_urls = _image_urls_for_product(app.state.conn, product_id)

    # Get description from SQLite
    desc_row = app.state.conn.execute(
        "SELECT description FROM products WHERE product_id = ?", (product_id,),
    ).fetchone()

    return ProductDetail(
        product_id=row["product_id"],
        merchant=row["merchant"],
        name=row["name"],
        price=row.get("price"),
        currency=row.get("currency"),
        url=row["url"],
        image_url=image_urls[0] if image_urls else None,
        description=desc_row[0] if desc_row else None,
        image_urls=image_urls,
    )


@app.get("/api/products/{product_id}/similar", response_model=SearchResponse)
def similar_products(
    product_id: str,
    top_k: int = Query(10, ge=1, le=50),
    merchant: Optional[str] = Query(None, description="Filter results by merchant"),
):
    """Find products similar to the given product using stored embeddings."""
    meta = app.state.meta
    idx_match = meta.index[meta["product_id"] == product_id]
    if idx_match.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    idx = idx_match[0]
    query_emb = app.state.matrix[idx : idx + 1]

    # Over-fetch to allow filtering
    fetch_k = min(top_k + 50, app.state.matrix.shape[0])
    results_df = search(query_emb, app.state.matrix, meta, top_k=fetch_k)

    # Exclude the query product itself
    results_df = results_df[results_df["product_id"] != product_id]

    if merchant:
        results_df = results_df[results_df["merchant"] == merchant]

    results_df = results_df.head(top_k)
    results = _df_to_search_results(results_df, app.state.conn)

    return SearchResponse(query_type="similar", results=results)


@app.get("/api/search/text", response_model=SearchResponse)
def text_search(
    q: str = Query(..., min_length=1, description="Text search query"),
    top_k: int = Query(10, ge=1, le=50),
    merchant: Optional[str] = Query(None, description="Filter results by merchant"),
):
    """Search products by text query using CLIP text encoder."""
    fetch_k = min(top_k + 50, app.state.matrix.shape[0])
    results_df = search_by_text(
        q, app.state.model, app.state.tokenizer,
        app.state.matrix, app.state.meta, app.state.device,
        top_k=fetch_k,
    )

    if merchant:
        results_df = results_df[results_df["merchant"] == merchant]

    results_df = results_df.head(top_k)
    results = _df_to_search_results(results_df, app.state.conn)

    return SearchResponse(query_type="text", results=results)


@app.post("/api/search/image", response_model=SearchResponse)
async def image_search(
    file: UploadFile,
    top_k: int = Query(10, ge=1, le=50),
    merchant: Optional[str] = Query(None, description="Filter results by merchant"),
):
    """Search products by uploading an image using CLIP image encoder."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()

        fetch_k = min(top_k + 50, app.state.matrix.shape[0])
        results_df = search_by_image(
            tmp.name, app.state.model, app.state.preprocess,
            app.state.matrix, app.state.meta, app.state.device,
            top_k=fetch_k,
        )

    if merchant:
        results_df = results_df[results_df["merchant"] == merchant]

    results_df = results_df.head(top_k)
    results = _df_to_search_results(results_df, app.state.conn)

    return SearchResponse(query_type="image", results=results)
