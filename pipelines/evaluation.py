"""Offline evaluation of recommendation models.

Five models evaluated with six IR metrics:
- Content-Based (CLIP): self-retrieval, relevance = same merchant
- Item-Based CF: leave-one-out on store level
- User-Based CF: leave-one-out on store level
- Random Baseline: uniform random sample
- Popularity Baseline: most-visited stores / most-common products
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ── Metric functions ────────────────────────────────────────────────────


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top = recommended[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in top-k."""
    if not relevant:
        return 0.0
    top = recommended[:k]
    return len(set(top) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    top = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if any relevant item appears in top-k, else 0."""
    top = recommended[:k]
    return 1.0 if set(top) & relevant else 0.0


def coverage(all_recommendations: list[list], catalog_size: int) -> float:
    """Fraction of catalog items appearing in any recommendation list."""
    if catalog_size == 0:
        return 0.0
    seen = set()
    for recs in all_recommendations:
        seen.update(recs)
    return len(seen) / catalog_size


def diversity(recommended: list, similarity_fn=None) -> float:
    """Average pairwise dissimilarity among recommended items.

    If no similarity_fn provided, returns 0.0 (used when we can't compute
    pairwise similarity, e.g. store-level models without embeddings).
    """
    if len(recommended) < 2 or similarity_fn is None:
        return 0.0
    n = len(recommended)
    total_dissim = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_fn(recommended[i], recommended[j])
            total_dissim += 1.0 - sim
            pairs += 1
    return total_dissim / pairs if pairs > 0 else 0.0


def compute_all_metrics(
    recommended: list, relevant: set, k: int, similarity_fn=None
) -> dict:
    """Compute all 6 metrics for a single recommendation list."""
    return {
        "precision": precision_at_k(recommended, relevant, k),
        "recall": recall_at_k(recommended, relevant, k),
        "ndcg": ndcg_at_k(recommended, relevant, k),
        "hit_rate": hit_rate_at_k(recommended, relevant, k),
    }


# ── Content-Based (CLIP) Evaluator ─────────────────────────────────────


def evaluate_content_based(matrix: np.ndarray, meta, k: int) -> dict:
    """Evaluate content-based CLIP model via self-retrieval.

    For each product, use its embedding as query, exclude self,
    check if same-merchant products appear in top-k.
    """
    n = matrix.shape[0]
    merchants = meta["merchant"].values
    all_recs = []
    metrics_list = []

    for i in range(n):
        query = matrix[i : i + 1]
        scores = (matrix @ query.T).flatten()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[::-1][:k]

        recommended = top_indices.tolist()
        # Relevant = other products from same merchant
        my_merchant = merchants[i]
        relevant = {j for j in range(n) if merchants[j] == my_merchant and j != i}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity: use embedding cosine similarity
    def sim_fn(a, b):
        return float(matrix[a] @ matrix[b])

    div_scores = []
    for recs in all_recs:
        if len(recs) >= 2:
            div_scores.append(diversity(recs, sim_fn))

    avg_metrics = {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
    }
    return avg_metrics


# ── Store-Level CF Evaluators ──────────────────────────────────────────


def _leave_one_out_store_splits(conn: sqlite3.Connection):
    """Generate leave-one-out splits for customers with 2+ store visits.

    Yields (customer_id, held_out_store, remaining_stores).
    """
    rows = conn.execute(
        "SELECT customer_id, store_id FROM transactions "
        "GROUP BY customer_id, store_id"
    ).fetchall()

    from collections import defaultdict

    customer_stores = defaultdict(set)
    for cid, sid in rows:
        customer_stores[cid].add(sid)

    for cid, stores in customer_stores.items():
        if len(stores) >= 2:
            stores_list = sorted(stores)
            for held_out in stores_list:
                remaining = stores - {held_out}
                yield cid, held_out, remaining


def evaluate_item_based_cf(
    sp_matrix: sparse.csr_matrix,
    sp_store_ids: list[str],
    conn: sqlite3.Connection,
    k: int,
) -> dict:
    """Evaluate item-based CF with leave-one-out.

    For each customer with 2+ store visits, hold out one store,
    use remaining stores to generate recommendations via item-based CF,
    check if held-out store appears in top-k.
    """
    store_idx_map = {s: i for i, s in enumerate(sp_store_ids)}
    n_stores = len(sp_store_ids)

    # Precompute store-store similarity matrix (work on copy)
    sim_matrix = cosine_similarity(sp_matrix)
    np.fill_diagonal(sim_matrix, 0)

    all_recs = []
    metrics_list = []

    for cid, held_out, remaining in _leave_one_out_store_splits(conn):
        if held_out not in store_idx_map:
            continue
        remaining_in_matrix = [s for s in remaining if s in store_idx_map]
        if not remaining_in_matrix:
            continue

        # Aggregate similarity scores from remaining stores
        store_scores = np.zeros(n_stores)
        for s in remaining_in_matrix:
            idx = store_idx_map[s]
            store_scores += sim_matrix[idx]

        # Zero out remaining stores (can't recommend already-visited)
        for s in remaining_in_matrix:
            store_scores[store_idx_map[s]] = -1

        top_indices = np.argsort(store_scores)[::-1][:k]
        recommended = [sp_store_ids[i] for i in top_indices]
        relevant = {held_out}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    # Diversity via store similarity
    def sim_fn(a, b):
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(sim_matrix[ia, ib])

    div_scores = [diversity(recs, sim_fn) for recs in all_recs if len(recs) >= 2]

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n_stores),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
    }


def evaluate_user_based_cf(
    cs_matrix: sparse.csr_matrix,
    cs_customer_ids: list[str],
    cs_store_ids: list[str],
    conn: sqlite3.Connection,
    k: int,
) -> dict:
    """Evaluate user-based CF with leave-one-out.

    For each customer with 2+ store visits, hold out one store,
    find similar customers, recommend their stores, check if held-out appears.
    """
    customer_idx_map = {c: i for i, c in enumerate(cs_customer_ids)}
    store_idx_map = {s: i for i, s in enumerate(cs_store_ids)}
    n_stores = len(cs_store_ids)

    # Precompute customer-customer similarity
    cust_sim = cosine_similarity(cs_matrix)
    np.fill_diagonal(cust_sim, 0)

    all_recs = []
    metrics_list = []

    for cid, held_out, remaining in _leave_one_out_store_splits(conn):
        if cid not in customer_idx_map or held_out not in store_idx_map:
            continue

        cidx = customer_idx_map[cid]

        # Build modified visit vector: remove held-out store
        held_out_sidx = store_idx_map[held_out]
        remaining_indices = {store_idx_map[s] for s in remaining if s in store_idx_map}
        if not remaining_indices:
            continue

        # Find top-20 similar customers
        sims = cust_sim[cidx]
        n_similar = min(20, len(cs_customer_ids) - 1)
        top_cust_indices = np.argsort(sims)[::-1][:n_similar]

        # Aggregate store scores from similar customers
        store_scores = np.zeros(n_stores)
        for ci in top_cust_indices:
            if sims[ci] <= 0:
                break
            their_stores = cs_matrix[ci].nonzero()[1]
            for si in their_stores:
                if si not in remaining_indices and si != held_out_sidx:
                    store_scores[si] += sims[ci]

        # Also zero out remaining (already visited)
        for si in remaining_indices:
            store_scores[si] = -1

        top_indices = np.argsort(store_scores)[::-1][:k]
        recommended = [cs_store_ids[i] for i in top_indices if store_scores[i] > 0]

        # Pad with empty if needed
        recommended = recommended[:k]
        relevant = {held_out}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    # Diversity: use customer-store matrix cosine similarity between stores
    sp_sim = None
    try:
        # Transpose cs_matrix to get store-customer, then compute store similarity
        store_features = cs_matrix.T.toarray()
        from sklearn.metrics.pairwise import cosine_similarity as cs
        sp_sim = cs(store_features)
    except Exception:
        pass

    def sim_fn(a, b):
        if sp_sim is None:
            return 0.0
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(sp_sim[ia, ib])

    div_scores = [diversity(recs, sim_fn) for recs in all_recs if len(recs) >= 2]

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n_stores),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
    }


# ── Baselines ──────────────────────────────────────────────────────────


def evaluate_random_baseline_products(matrix: np.ndarray, meta, k: int) -> dict:
    """Random baseline for product-level: uniform random sample of k products."""
    rng = np.random.RandomState(42)
    n = matrix.shape[0]
    merchants = meta["merchant"].values
    all_recs = []
    metrics_list = []

    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        recommended = rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()
        my_merchant = merchants[i]
        relevant = {j for j in range(n) if merchants[j] == my_merchant and j != i}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": 1.0,  # random selections are maximally diverse
    }


def evaluate_random_baseline_stores(
    conn: sqlite3.Connection, store_ids: list[str], k: int
) -> dict:
    """Random baseline for store-level: uniform random store sample."""
    rng = np.random.RandomState(42)
    n_stores = len(store_ids)
    store_set = set(store_ids)
    all_recs = []
    metrics_list = []

    for cid, held_out, remaining in _leave_one_out_store_splits(conn):
        if held_out not in store_set:
            continue
        excluded = remaining | {held_out}
        candidates = [s for s in store_ids if s not in excluded]
        if not candidates:
            continue
        recommended = list(rng.choice(candidates, size=min(k, len(candidates)), replace=False))
        relevant = {held_out}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n_stores),
        "diversity": 1.0,
    }


def evaluate_popularity_baseline_products(matrix: np.ndarray, meta, k: int) -> dict:
    """Popularity baseline: always recommend the most common merchant's products.

    For each product, recommend the k products from the largest merchant group.
    """
    n = matrix.shape[0]
    merchants = meta["merchant"].values

    # Count products per merchant, find most popular
    from collections import Counter
    merchant_counts = Counter(merchants)
    most_popular_merchant = merchant_counts.most_common(1)[0][0]

    # Products from most popular merchant
    popular_indices = [i for i in range(n) if merchants[i] == most_popular_merchant]

    all_recs = []
    metrics_list = []

    for i in range(n):
        # Recommend top-k from most popular merchant (excluding self)
        candidates = [j for j in popular_indices if j != i][:k]
        recommended = candidates

        my_merchant = merchants[i]
        relevant = {j for j in range(n) if merchants[j] == my_merchant and j != i}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": 0.0,  # always same items = no diversity
    }


def evaluate_popularity_baseline_stores(
    conn: sqlite3.Connection, store_ids: list[str], k: int
) -> dict:
    """Popularity baseline: always recommend most-visited stores."""
    # Get store visit counts
    rows = conn.execute(
        "SELECT store_id, COUNT(DISTINCT customer_id) as n_customers "
        "FROM transactions GROUP BY store_id ORDER BY n_customers DESC"
    ).fetchall()
    popular_stores = [r[0] for r in rows if r[0] in set(store_ids)][:k * 2]

    store_set = set(store_ids)
    all_recs = []
    metrics_list = []

    for cid, held_out, remaining in _leave_one_out_store_splits(conn):
        if held_out not in store_set:
            continue
        excluded = remaining | {held_out}
        recommended = [s for s in popular_stores if s not in excluded][:k]
        relevant = {held_out}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, len(store_ids)),
        "diversity": 0.0,
    }


# ── Master Evaluation Function ─────────────────────────────────────────


def run_full_evaluation(
    matrix: np.ndarray,
    meta,
    sp_matrix: Optional[sparse.csr_matrix],
    sp_store_ids: Optional[list[str]],
    cs_matrix: Optional[sparse.csr_matrix],
    cs_customer_ids: Optional[list[str]],
    cs_store_ids: Optional[list[str]],
    conn: sqlite3.Connection,
    k: int = 5,
) -> dict:
    """Run all 5 model evaluations and return results dict.

    Returns: {
        "k": k,
        "models": [
            {"name": "Content-Based (CLIP)", "level": "Product", "metrics": {...}},
            ...
        ]
    }
    """
    models = []

    # 1. Content-Based (CLIP)
    cb_metrics = evaluate_content_based(matrix, meta, k)
    models.append({
        "name": "Content-Based (CLIP)",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in cb_metrics.items()},
    })

    # 2. Item-Based CF
    if sp_matrix is not None:
        ib_metrics = evaluate_item_based_cf(sp_matrix, sp_store_ids, conn, k)
        models.append({
            "name": "Item-Based CF",
            "level": "Store",
            "metrics": {key: round(float(val), 4) for key, val in ib_metrics.items()},
        })

    # 3. User-Based CF
    if cs_matrix is not None:
        ub_metrics = evaluate_user_based_cf(
            cs_matrix, cs_customer_ids, cs_store_ids, conn, k
        )
        models.append({
            "name": "User-Based CF",
            "level": "Store",
            "metrics": {key: round(float(val), 4) for key, val in ub_metrics.items()},
        })

    # 4. Random Baseline (product + store)
    rand_product = evaluate_random_baseline_products(matrix, meta, k)
    models.append({
        "name": "Random Baseline",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in rand_product.items()},
    })

    if cs_store_ids:
        rand_store = evaluate_random_baseline_stores(conn, cs_store_ids, k)
        # Merge: average product + store baselines
        merged = {}
        for m_key in rand_product:
            merged[m_key] = round((rand_product[m_key] + rand_store[m_key]) / 2, 4)
        models[-1]["level"] = "Both"
        models[-1]["metrics"] = merged

    # 5. Popularity Baseline (product + store)
    pop_product = evaluate_popularity_baseline_products(matrix, meta, k)
    models.append({
        "name": "Popularity Baseline",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in pop_product.items()},
    })

    if cs_store_ids:
        pop_store = evaluate_popularity_baseline_stores(conn, cs_store_ids, k)
        merged = {}
        for m_key in pop_product:
            merged[m_key] = round((pop_product[m_key] + pop_store[m_key]) / 2, 4)
        models[-1]["level"] = "Both"
        models[-1]["metrics"] = merged

    return {"k": k, "models": models}
