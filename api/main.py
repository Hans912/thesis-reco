"""FastAPI recommendation API — loads CLIP model once, serves similarity search."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import base64
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.models import (
    ChatRequest,
    ChatResponse,
    EvaluationResults,
    FavoriteItem,
    FavoriteRequest,
    FavoritesResponse,
    ProductDetail,
    ProductListResponse,
    ProductSummary,
    Profile,
    SearchResponse,
    SearchResult,
    Store,
    StoreListResponse,
    StoreRecommendation,
    StoreRecommendationResponse,
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

    # Load collaborative filtering matrices if transaction data exists
    try:
        from pipelines.collab import build_store_product_matrix, build_customer_store_matrix

        sp_matrix, sp_store_ids, sp_product_ids = build_store_product_matrix(conn)
        cs_matrix, cs_customer_ids, cs_store_ids = build_customer_store_matrix(conn)
        app.state.sp_matrix = sp_matrix
        app.state.sp_store_ids = sp_store_ids
        app.state.cs_matrix = cs_matrix
        app.state.cs_customer_ids = cs_customer_ids
        app.state.cs_store_ids = cs_store_ids
        print(f"Collab matrices loaded: {sp_matrix.shape[0]} stores x {sp_matrix.shape[1]} products, "
              f"{cs_matrix.shape[0]} customers x {cs_matrix.shape[1]} stores", flush=True)
    except Exception as e:
        print(f"Collab filtering not available: {e}", flush=True)
        app.state.sp_matrix = None
        app.state.cs_matrix = None

    yield

    conn.close()


app = FastAPI(title="Thesis Reco API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Conversational product recommendation via GPT-4o-mini with tool calling."""
    from api.chat import chat

    image_bytes = None
    if request.image:
        try:
            image_bytes = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = await chat(messages, image_bytes, app.state)

    return ChatResponse(
        message=result["message"],
        products=result.get("products"),
        follow_up_questions=result.get("follow_up_questions"),
        stores=result.get("stores"),
    )


@app.get("/api/stores", response_model=StoreListResponse)
def list_stores(
    merchant: Optional[str] = Query(None, description="Filter by merchant"),
):
    """List store locations, optionally filtered by merchant."""
    conn = app.state.conn

    # Check if stores table exists
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stores'"
    ).fetchone()
    if not table_check:
        return StoreListResponse(stores=[])

    if merchant:
        rows = conn.execute(
            "SELECT store_id, merchant, display_name, lat, lng, street, street_number, zip_code, google_place_id "
            "FROM stores WHERE merchant = ? ORDER BY display_name",
            (merchant,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT store_id, merchant, display_name, lat, lng, street, street_number, zip_code, google_place_id "
            "FROM stores ORDER BY merchant, display_name",
        ).fetchall()

    stores = [
        Store(
            store_id=r[0], merchant=r[1], display_name=r[2],
            lat=r[3], lng=r[4], street=r[5], street_number=r[6],
            zip_code=r[7], google_place_id=r[8],
        )
        for r in rows
    ]
    return StoreListResponse(stores=stores)


# ── Favorites ───────────────────────────────────────────────────────────

@app.get("/api/favorites", response_model=FavoritesResponse)
def get_favorites(session_id: str = Query(..., description="Session or profile ID")):
    """List all favorited products for a session."""
    conn = app.state.conn
    rows = conn.execute(
        """SELECT f.product_id, f.added_at, p.merchant, p.name, p.price, p.currency, p.url
           FROM favorites f
           JOIN products p ON f.product_id = p.product_id
           WHERE f.session_id = ?
           ORDER BY f.added_at DESC""",
        (session_id,),
    ).fetchall()

    items = []
    for r in rows:
        img = _first_image_url(conn, r[0])
        items.append(FavoriteItem(
            product_id=r[0], added_at=r[1], merchant=r[2], name=r[3],
            price=r[4], currency=r[5], url=r[6], image_url=img,
        ))
    return FavoritesResponse(favorites=items)


@app.post("/api/favorites", status_code=201)
def add_favorite(req: FavoriteRequest):
    """Add a product to favorites."""
    conn = app.state.conn
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO favorites (session_id, product_id, added_at) VALUES (?, ?, ?)",
            (req.session_id, req.product_id, now),
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@app.delete("/api/favorites/{product_id}")
def remove_favorite(
    product_id: str,
    session_id: str = Query(..., description="Session or profile ID"),
):
    """Remove a product from favorites."""
    conn = app.state.conn
    conn.execute(
        "DELETE FROM favorites WHERE session_id = ? AND product_id = ?",
        (session_id, product_id),
    )
    conn.commit()
    return {"ok": True}


# ── Collaborative Filtering ─────────────────────────────────────────────

@app.get("/api/stores/{store_id}/similar", response_model=StoreRecommendationResponse)
def get_similar_stores(store_id: str, top_k: int = Query(5, ge=1, le=20)):
    """Item-based CF: find stores with a similar product mix."""
    if app.state.sp_matrix is None:
        raise HTTPException(status_code=503, detail="Collaborative filtering not available")

    from pipelines.collab import similar_stores

    results = similar_stores(
        store_id, app.state.sp_matrix, app.state.sp_store_ids,
        app.state.conn, top_k=top_k,
    )
    return StoreRecommendationResponse(
        method="item_based",
        results=[StoreRecommendation(**r) for r in results],
    )


@app.get("/api/recommend/stores", response_model=StoreRecommendationResponse)
def recommend_stores(
    customer_id: str = Query(None, description="Customer ID from profile"),
    profile_id: str = Query(None, description="Profile ID to look up customer_id"),
    top_k: int = Query(5, ge=1, le=20),
):
    """User-based CF: recommend stores based on co-shopping patterns."""
    if app.state.cs_matrix is None:
        raise HTTPException(status_code=503, detail="Collaborative filtering not available")

    # Resolve customer_id from profile if needed
    if not customer_id and profile_id:
        profiles = _load_profiles()
        for p in profiles:
            if p["profile_id"] == profile_id:
                customer_id = p.get("customer_id")
                break

    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id or valid profile_id required")

    from pipelines.collab import recommend_stores_for_customer

    results = recommend_stores_for_customer(
        customer_id, app.state.cs_matrix, app.state.cs_customer_ids,
        app.state.cs_store_ids, app.state.conn, top_k=top_k,
    )
    return StoreRecommendationResponse(
        method="user_based",
        results=[StoreRecommendation(**r) for r in results],
    )


# ── Evaluation ──────────────────────────────────────────────────────────

@app.get("/api/evaluation/results", response_model=EvaluationResults)
def evaluation_results(k: int = Query(5, ge=1, le=20)):
    """Run offline evaluation of all 5 recommendation models."""
    # Check cache
    cache = getattr(app.state, "_eval_cache", {})
    if k in cache:
        return cache[k]

    from pipelines.evaluation import run_full_evaluation

    results = run_full_evaluation(
        matrix=app.state.matrix,
        meta=app.state.meta,
        sp_matrix=app.state.sp_matrix,
        sp_store_ids=getattr(app.state, "sp_store_ids", None),
        cs_matrix=app.state.cs_matrix,
        cs_customer_ids=getattr(app.state, "cs_customer_ids", None),
        cs_store_ids=getattr(app.state, "cs_store_ids", None),
        conn=app.state.conn,
        k=k,
    )

    # Cache results
    if not hasattr(app.state, "_eval_cache"):
        app.state._eval_cache = {}
    app.state._eval_cache[k] = results

    return results


# ── Profiles ────────────────────────────────────────────────────────────

import json as _json

_PROFILES_PATH = ROOT / "data" / "mock_profiles.json"


def _load_profiles() -> list[dict]:
    if _PROFILES_PATH.exists():
        with open(_PROFILES_PATH) as f:
            return _json.load(f)
    return []


@app.get("/api/profiles", response_model=list[Profile])
def list_profiles():
    """List available user profiles for the CF demo."""
    return [Profile(**p) for p in _load_profiles()]


@app.get("/api/profiles/{profile_id}/top-store")
def profile_top_store(profile_id: str):
    """Return the most-visited store for a profile's customer."""
    profiles = _load_profiles()
    customer_id = None
    for p in profiles:
        if p["profile_id"] == profile_id:
            customer_id = p.get("customer_id")
            break
    if not customer_id:
        raise HTTPException(status_code=404, detail="Profile not found")

    row = app.state.conn.execute(
        "SELECT store_id, COUNT(*) as cnt FROM transactions "
        "WHERE customer_id = ? GROUP BY store_id ORDER BY cnt DESC LIMIT 1",
        (customer_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No transactions for this profile")

    profile_row = app.state.conn.execute(
        "SELECT merchant_name, city FROM store_profiles WHERE store_id = ?",
        (row[0],),
    ).fetchone()

    return {
        "store_id": row[0],
        "visit_count": row[1],
        "merchant_name": profile_row[0] if profile_row else None,
        "city": profile_row[1] if profile_row else None,
    }
