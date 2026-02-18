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
