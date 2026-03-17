"""Pydantic request/response schemas for the recommendation API."""

from typing import List, Optional

from pydantic import BaseModel


class ProductSummary(BaseModel):
    product_id: str
    merchant: str
    name: str
    price: Optional[str] = None
    currency: Optional[str] = None
    url: str
    image_url: Optional[str] = None  # first image thumbnail


class ProductDetail(ProductSummary):
    description: Optional[str] = None
    image_urls: List[str] = []


class SearchResult(ProductSummary):
    score: float


class SearchResponse(BaseModel):
    query_type: str  # "text", "image", "similar"
    results: List[SearchResult]


class ProductListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    products: List[ProductSummary]


# ── Chat models ──────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    image: Optional[str] = None  # base64-encoded image, or null
    city: Optional[str] = None   # city filter for product results


class ChatProduct(BaseModel):
    product_id: str
    merchant: str
    name: str
    price: Optional[str] = None
    currency: Optional[str] = None
    url: str
    image_url: Optional[str] = None
    score: Optional[float] = None


class ChatStore(BaseModel):
    store_id: str
    merchant: str
    display_name: Optional[str] = None
    lat: float
    lng: float
    address: Optional[str] = None
    distance_km: Optional[float] = None


class ChatResponse(BaseModel):
    message: str
    products: Optional[List[ChatProduct]] = None
    follow_up_questions: Optional[List[str]] = None
    stores: Optional[List[ChatStore]] = None


# ── Store models ─────────────────────────────────────────────────────────


class Store(BaseModel):
    store_id: str
    merchant: str
    display_name: Optional[str] = None
    lat: float
    lng: float
    street: Optional[str] = None
    street_number: Optional[str] = None
    zip_code: Optional[str] = None
    google_place_id: Optional[str] = None


class StoreListResponse(BaseModel):
    stores: List[Store]


# ── Favorites models ─────────────────────────────────────────────────────


class FavoriteRequest(BaseModel):
    session_id: str
    product_id: str


class FavoriteItem(ProductSummary):
    added_at: str


class FavoritesResponse(BaseModel):
    favorites: List[FavoriteItem]


# ── Collaborative filtering models ───────────────────────────────────────


class StoreRecommendation(BaseModel):
    store_id: str
    merchant_name: Optional[str] = None
    city: Optional[str] = None
    score: float
    num_products: Optional[int] = None
    median_price: Optional[float] = None


class StoreRecommendationResponse(BaseModel):
    method: str  # "item_based" or "user_based"
    results: List[StoreRecommendation]


class Profile(BaseModel):
    profile_id: str
    name: str
    description: str
    emoji: str
    customer_id: Optional[str] = None


# ── Evaluation models ───────────────────────────────────────────────────


class ModelMetrics(BaseModel):
    precision: float
    recall: float
    ndcg: float
    hit_rate: float
    coverage: float
    diversity: float
    novelty: Optional[float] = None  # product-level models only


class ModelResult(BaseModel):
    name: str
    level: str  # "Product", "Store", or "Both"
    metrics: ModelMetrics
    stats: Optional[dict] = None  # optional extra info (e.g. n_categorized for CLIP)


class DataStats(BaseModel):
    n_customers_train: int
    n_stores_train: int
    n_products: int
    n_train_transactions: int
    n_test_transactions: int
    date_range_min: Optional[str] = None
    date_range_max: Optional[str] = None
    split_date: str
    n_test_cases: int
    sparsity_customer_store: float
    sparsity_store_product: float
    avg_stores_per_customer: float
    avg_products_per_store: float
    n_embeddings: int
    embedding_dim: int


class EvaluationResults(BaseModel):
    k: int
    split_date: Optional[str] = None
    n_test_cases: Optional[int] = None
    data_stats: Optional[DataStats] = None
    models: List[ModelResult]
