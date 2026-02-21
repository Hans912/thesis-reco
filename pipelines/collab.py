"""Store-based collaborative filtering using transaction data.

Two approaches:
1. Item-based: cosine similarity between store-product vectors
   ("stores with a similar product mix")
2. User-based: cosine similarity between customer-store visit vectors
   ("customers who shop at similar stores")
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def build_store_product_matrix(conn: sqlite3.Connection):
    """Build a sparse store x product matrix from transactions.

    Values = total quantity sold of each product at each store.
    Returns (matrix, store_ids, product_ids).
    """
    rows = conn.execute(
        "SELECT store_id, product_id, SUM(qty) as total_qty "
        "FROM transactions GROUP BY store_id, product_id"
    ).fetchall()

    store_set = sorted(set(r[0] for r in rows))
    product_set = sorted(set(r[1] for r in rows))
    store_idx = {s: i for i, s in enumerate(store_set)}
    product_idx = {p: i for i, p in enumerate(product_set)}

    row_indices = []
    col_indices = []
    values = []
    for store_id, product_id, qty in rows:
        row_indices.append(store_idx[store_id])
        col_indices.append(product_idx[product_id])
        values.append(qty or 1.0)

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(store_set), len(product_set)),
    )
    return matrix, store_set, product_set


def build_customer_store_matrix(conn: sqlite3.Connection):
    """Build a sparse customer x store matrix from transactions.

    Values = total spend at each store.
    Returns (matrix, customer_ids, store_ids).
    """
    rows = conn.execute(
        "SELECT customer_id, store_id, SUM(unit_price * qty) as spend "
        "FROM transactions GROUP BY customer_id, store_id"
    ).fetchall()

    customer_set = sorted(set(r[0] for r in rows))
    store_set = sorted(set(r[1] for r in rows))
    customer_idx = {c: i for i, c in enumerate(customer_set)}
    store_idx = {s: i for i, s in enumerate(store_set)}

    row_indices = []
    col_indices = []
    values = []
    for customer_id, store_id, spend in rows:
        row_indices.append(customer_idx[customer_id])
        col_indices.append(store_idx[store_id])
        values.append(spend if spend else 1.0)

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(customer_set), len(store_set)),
    )
    return matrix, customer_set, store_set


def similar_stores(
    store_id: str,
    matrix: sparse.csr_matrix,
    store_ids: list[str],
    conn: sqlite3.Connection,
    top_k: int = 5,
) -> list[dict]:
    """Find stores with the most similar product mix (item-based CF).

    Returns list of dicts with store_id, score, and profile info.
    """
    if store_id not in store_ids:
        return []

    idx = store_ids.index(store_id)
    query_vec = matrix[idx]
    sims = cosine_similarity(query_vec, matrix).flatten()

    # Exclude self
    sims[idx] = -1
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_indices:
        if sims[i] <= 0:
            break
        sid = store_ids[i]
        profile = _get_store_profile(conn, sid)
        results.append({
            "store_id": sid,
            "score": round(float(sims[i]), 4),
            **profile,
        })
    return results


def recommend_stores_for_customer(
    customer_id: str,
    matrix: sparse.csr_matrix,
    customer_ids: list[str],
    store_ids: list[str],
    conn: sqlite3.Connection,
    top_k: int = 5,
) -> list[dict]:
    """Recommend stores for a customer based on co-shopping patterns (user-based CF).

    Finds similar customers, then recommends stores they visited that this customer hasn't.
    """
    if customer_id not in customer_ids:
        return []

    idx = customer_ids.index(customer_id)
    query_vec = matrix[idx]

    # Find similar customers
    sims = cosine_similarity(query_vec, matrix).flatten()
    sims[idx] = -1  # exclude self

    # Stores this customer already visited
    visited = set(matrix[idx].nonzero()[1])

    # Weighted score for each unvisited store from similar customers
    n_similar = min(20, len(customer_ids) - 1)
    top_customer_indices = np.argsort(sims)[::-1][:n_similar]

    store_scores = {}
    for ci in top_customer_indices:
        if sims[ci] <= 0:
            break
        customer_stores = matrix[ci].nonzero()[1]
        for si in customer_stores:
            if si not in visited:
                store_scores[si] = store_scores.get(si, 0) + sims[ci]

    # Sort by aggregated score
    ranked = sorted(store_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for si, score in ranked:
        sid = store_ids[si]
        profile = _get_store_profile(conn, sid)
        results.append({
            "store_id": sid,
            "score": round(float(score), 4),
            **profile,
        })
    return results


def _get_store_profile(conn: sqlite3.Connection, store_id: str) -> dict:
    """Fetch store profile metadata from store_profiles table."""
    row = conn.execute(
        "SELECT merchant_name, city, num_invoices, num_distinct_products, median_unit_price "
        "FROM store_profiles WHERE store_id = ?",
        (store_id,),
    ).fetchone()
    if row:
        return {
            "merchant_name": row[0],
            "city": row[1],
            "num_invoices": row[2],
            "num_products": row[3],
            "median_price": row[4],
        }
    return {}
