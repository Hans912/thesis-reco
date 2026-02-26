"""Offline evaluation of recommendation models.

Uses a temporal train/test split to avoid data leakage:
- Train: all transactions before the split date
- Test: customers who visited NEW stores after the split date
- Models are trained only on train data, then evaluated on test

Eight models evaluated with six IR metrics:
- Content-Based (CLIP): self-retrieval, relevance = same merchant
- Item-Based CF (memory): temporal split on store level
- User-Based CF (memory): temporal split on store level
- ALS (model-based): implicit feedback matrix factorization
- LightFM WARP: pairwise ranking loss, no side features
- LightFM WARP + Features: hybrid model with store profile features
- Random Baseline: uniform random sample
- Popularity Baseline: most-visited stores / most-common products
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ── Default split date ─────────────────────────────────────────────────

SPLIT_DATE = "2025-12-01"


# ── Metric functions ────────────────────────────────────────────────────
# Standard IR metrics following established formulas:
# - Precision@K, Recall@K: Manning et al., Introduction to IR, 2008
# - nDCG@K: Järvelin & Kekäläinen, ACM TOIS, 2002
# - Hit Rate@K: Deshpande & Karypis, ACM TOIS, 2004
# - Coverage: Adomavicius & Kwon, IEEE TKDE, 2012
# - Diversity (ILD): Ziegler et al., WWW, 2005


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
    """Average pairwise dissimilarity among recommended items."""
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
    recommended: list, relevant: set, k: int,
) -> dict:
    """Compute precision, recall, nDCG, hit_rate for a single recommendation list."""
    return {
        "precision": precision_at_k(recommended, relevant, k),
        "recall": recall_at_k(recommended, relevant, k),
        "ndcg": ndcg_at_k(recommended, relevant, k),
        "hit_rate": hit_rate_at_k(recommended, relevant, k),
    }


# ── Temporal Train/Test Split ──────────────────────────────────────────


def build_temporal_test_cases(conn: sqlite3.Connection, split_date: str = SPLIT_DATE):
    """Build test cases from temporal split.

    Returns list of (customer_id, train_stores, test_new_stores) where
    test_new_stores are stores visited AFTER split_date that were NOT
    visited before it.
    """
    # Train: customer-store pairs before split
    train_rows = conn.execute(
        "SELECT customer_id, store_id FROM transactions "
        "WHERE issued_on < ? GROUP BY customer_id, store_id",
        (split_date,),
    ).fetchall()

    train_map = defaultdict(set)
    for cid, sid in train_rows:
        train_map[cid].add(sid)

    # Test: customer-store pairs after split
    test_rows = conn.execute(
        "SELECT customer_id, store_id FROM transactions "
        "WHERE issued_on >= ? GROUP BY customer_id, store_id",
        (split_date,),
    ).fetchall()

    test_map = defaultdict(set)
    for cid, sid in test_rows:
        test_map[cid].add(sid)

    # Build test cases: customers in both with new stores
    test_cases = []
    for cid in train_map.keys() & test_map.keys():
        new_stores = test_map[cid] - train_map[cid]
        if new_stores:
            test_cases.append((cid, train_map[cid], new_stores))

    return test_cases


def build_train_store_product_matrix(conn: sqlite3.Connection, split_date: str = SPLIT_DATE):
    """Build store-product matrix from TRAIN data only."""
    rows = conn.execute(
        "SELECT store_id, product_id, SUM(qty) as total_qty "
        "FROM transactions WHERE issued_on < ? "
        "GROUP BY store_id, product_id",
        (split_date,),
    ).fetchall()

    store_set = sorted(set(r[0] for r in rows))
    product_set = sorted(set(r[1] for r in rows))
    store_idx = {s: i for i, s in enumerate(store_set)}
    product_idx = {p: i for i, p in enumerate(product_set)}

    row_indices, col_indices, values = [], [], []
    for store_id, product_id, qty in rows:
        row_indices.append(store_idx[store_id])
        col_indices.append(product_idx[product_id])
        values.append(float(qty))

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(store_set), len(product_set)),
    )
    return matrix, store_set, product_set


def build_train_customer_store_matrix(conn: sqlite3.Connection, split_date: str = SPLIT_DATE):
    """Build customer-store matrix from TRAIN data only."""
    rows = conn.execute(
        "SELECT customer_id, store_id, SUM(qty * unit_price) as spend "
        "FROM transactions WHERE issued_on < ? "
        "GROUP BY customer_id, store_id",
        (split_date,),
    ).fetchall()

    customer_set = sorted(set(r[0] for r in rows))
    store_set = sorted(set(r[1] for r in rows))
    customer_idx = {c: i for i, c in enumerate(customer_set)}
    store_idx = {s: i for i, s in enumerate(store_set)}

    row_indices, col_indices, values = [], [], []
    for cid, sid, spend in rows:
        row_indices.append(customer_idx[cid])
        col_indices.append(store_idx[sid])
        values.append(float(spend))

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(customer_set), len(store_set)),
    )
    return matrix, customer_set, store_set


def build_train_interaction_matrix(conn: sqlite3.Connection, split_date: str = SPLIT_DATE):
    """Build recency-weighted customer-store interaction matrix from TRAIN data only.

    Used by model-based CF (ALS, LightFM). Uses shared builder from collab_model.
    """
    from pipelines.collab_model import _build_interaction_matrix
    return _build_interaction_matrix(conn, before_date=split_date)


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
        my_merchant = merchants[i]
        relevant = {j for j in range(n) if merchants[j] == my_merchant and j != i}

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity: use embedding cosine similarity
    def sim_fn(a, b):
        return float(matrix[a] @ matrix[b])

    div_scores = [diversity(recs, sim_fn) for recs in all_recs if len(recs) >= 2]

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
    }


# ── Store-Level CF Evaluators (Temporal) ──────────────────────────────


def _aggregate_metrics(metrics_list, all_recs, n_stores, sim_fn=None):
    """Aggregate per-test-case metrics into model-level averages."""
    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    div_scores = [diversity(recs, sim_fn) for recs in all_recs if len(recs) >= 2] if sim_fn else []

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n_stores),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
    }


def evaluate_item_based_cf(
    sp_matrix: sparse.csr_matrix,
    sp_store_ids: list[str],
    test_cases: list,
    k: int,
) -> dict:
    """Evaluate item-based CF with temporal split.

    For each test case, use train stores to find similar stores,
    check if any test new stores appear in top-k.
    """
    store_idx_map = {s: i for i, s in enumerate(sp_store_ids)}
    n_stores = len(sp_store_ids)

    sim_matrix = cosine_similarity(sp_matrix)
    np.fill_diagonal(sim_matrix, 0)

    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        train_in_matrix = [s for s in train_stores if s in store_idx_map]
        if not train_in_matrix:
            continue

        # Aggregate similarity scores from train stores
        store_scores = np.zeros(n_stores)
        for s in train_in_matrix:
            store_scores += sim_matrix[store_idx_map[s]]

        # Zero out train stores (can't recommend already-visited)
        for s in train_in_matrix:
            store_scores[store_idx_map[s]] = -1

        top_indices = np.argsort(store_scores)[::-1][:k]
        recommended = [sp_store_ids[i] for i in top_indices]
        relevant = {s for s in test_new_stores if s in store_idx_map}
        if not relevant:
            continue

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    def sim_fn(a, b):
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(sim_matrix[ia, ib])

    return _aggregate_metrics(metrics_list, all_recs, n_stores, sim_fn)


def evaluate_user_based_cf(
    cs_matrix: sparse.csr_matrix,
    cs_customer_ids: list[str],
    cs_store_ids: list[str],
    test_cases: list,
    k: int,
) -> dict:
    """Evaluate user-based CF with temporal split.

    For each test case, find similar customers from train data,
    recommend their stores, check if test new stores appear.
    """
    customer_idx_map = {c: i for i, c in enumerate(cs_customer_ids)}
    store_idx_map = {s: i for i, s in enumerate(cs_store_ids)}
    n_stores = len(cs_store_ids)

    cust_sim = cosine_similarity(cs_matrix)
    np.fill_diagonal(cust_sim, 0)

    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        if cid not in customer_idx_map:
            continue

        cidx = customer_idx_map[cid]
        train_indices = {store_idx_map[s] for s in train_stores if s in store_idx_map}
        if not train_indices:
            continue

        relevant = {s for s in test_new_stores if s in store_idx_map}
        if not relevant:
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
                if si not in train_indices:
                    store_scores[si] += sims[ci]

        # Zero out train stores
        for si in train_indices:
            store_scores[si] = -1

        top_indices = np.argsort(store_scores)[::-1][:k]
        recommended = [cs_store_ids[i] for i in top_indices if store_scores[i] > 0][:k]

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity via store similarity from customer-store matrix
    sp_sim = None
    try:
        store_features = cs_matrix.T.toarray()
        sp_sim = cosine_similarity(store_features)
    except Exception:
        pass

    def sim_fn(a, b):
        if sp_sim is None:
            return 0.0
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(sp_sim[ia, ib])

    return _aggregate_metrics(metrics_list, all_recs, n_stores, sim_fn)


# ── Model-Based CF Evaluators (Temporal) ──────────────────────────────


def evaluate_als(
    als_model,
    interaction_matrix: sparse.csr_matrix,
    customer_ids: list[str],
    store_ids: list[str],
    test_cases: list,
    k: int,
) -> dict:
    """Evaluate ALS model with temporal split."""
    customer_idx_map = {c: i for i, c in enumerate(customer_ids)}
    store_idx_map = {s: i for i, s in enumerate(store_ids)}
    n_stores = len(store_ids)

    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        if cid not in customer_idx_map:
            continue

        cidx = customer_idx_map[cid]
        relevant = {s for s in test_new_stores if s in store_idx_map}
        if not relevant:
            continue

        try:
            ids, scores = als_model.recommend(
                cidx, interaction_matrix[cidx], N=k,
                filter_already_liked_items=True,
            )
            rec_ids = ids.tolist()
        except Exception:
            continue

        recommended = [store_ids[i] for i in rec_ids]

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity via item factors
    item_factors = als_model.item_factors
    if hasattr(item_factors, 'to_numpy'):
        item_factors = item_factors.to_numpy()
    item_factors = np.array(item_factors)

    def sim_fn(a, b):
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        va, vb = item_factors[ia], item_factors[ib]
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(va @ vb / norm) if norm > 0 else 0.0

    return _aggregate_metrics(metrics_list, all_recs, n_stores, sim_fn)


def evaluate_lightfm(
    lfm_model,
    interaction_matrix: sparse.csr_matrix,
    customer_ids: list[str],
    store_ids: list[str],
    test_cases: list,
    k: int,
    item_features=None,
    user_features=None,
) -> dict:
    """Evaluate a LightFM model with temporal split."""
    customer_idx_map = {c: i for i, c in enumerate(customer_ids)}
    store_idx_map = {s: i for i, s in enumerate(store_ids)}
    n_stores = len(store_ids)

    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        if cid not in customer_idx_map:
            continue

        cidx = customer_idx_map[cid]
        relevant = {s for s in test_new_stores if s in store_idx_map}
        if not relevant:
            continue

        try:
            scores = lfm_model.predict(
                cidx, np.arange(n_stores),
                item_features=item_features, user_features=user_features,
            )
            # Filter out stores already visited in train
            visited = set(interaction_matrix[cidx].nonzero()[1])
            for v in visited:
                scores[v] = -np.inf
            top_indices = np.argsort(-scores)[:k]
            rec_ids = top_indices.tolist()
        except Exception:
            continue

        recommended = [store_ids[i] for i in rec_ids]

        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity via LightFM item embeddings
    _, item_embeddings = lfm_model.get_item_representations(features=item_features)

    def sim_fn(a, b):
        ia, ib = store_idx_map.get(a), store_idx_map.get(b)
        if ia is None or ib is None:
            return 0.0
        va, vb = item_embeddings[ia], item_embeddings[ib]
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(va @ vb / norm) if norm > 0 else 0.0

    return _aggregate_metrics(metrics_list, all_recs, n_stores, sim_fn)


# ── Baselines (Temporal) ──────────────────────────────────────────────


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
        "diversity": 1.0,
    }


def evaluate_random_baseline_stores(
    store_ids: list[str], test_cases: list, k: int,
) -> dict:
    """Random baseline for store-level: uniform random store sample."""
    rng = np.random.RandomState(42)
    n_stores = len(store_ids)
    store_set = set(store_ids)
    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        candidates = [s for s in store_ids if s not in train_stores]
        relevant = {s for s in test_new_stores if s in store_set}
        if not candidates or not relevant:
            continue
        recommended = list(rng.choice(candidates, size=min(k, len(candidates)), replace=False))

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
    """Popularity baseline: always recommend the most common merchant's products."""
    n = matrix.shape[0]
    merchants = meta["merchant"].values

    from collections import Counter
    merchant_counts = Counter(merchants)
    most_popular_merchant = merchant_counts.most_common(1)[0][0]
    popular_indices = [i for i in range(n) if merchants[i] == most_popular_merchant]

    all_recs = []
    metrics_list = []

    for i in range(n):
        candidates = [j for j in popular_indices if j != i][:k]
        my_merchant = merchants[i]
        relevant = {j for j in range(n) if merchants[j] == my_merchant and j != i}

        m = compute_all_metrics(candidates, relevant, k)
        metrics_list.append(m)
        all_recs.append(candidates)

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": 0.0,
    }


def evaluate_popularity_baseline_stores(
    conn: sqlite3.Connection,
    store_ids: list[str],
    test_cases: list,
    k: int,
    split_date: str = SPLIT_DATE,
) -> dict:
    """Popularity baseline: always recommend most-visited stores (from train period)."""
    rows = conn.execute(
        "SELECT store_id, COUNT(DISTINCT customer_id) as n_customers "
        "FROM transactions WHERE issued_on < ? "
        "GROUP BY store_id ORDER BY n_customers DESC",
        (split_date,),
    ).fetchall()
    popular_stores = [r[0] for r in rows if r[0] in set(store_ids)][:k * 2]

    store_set = set(store_ids)
    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        recommended = [s for s in popular_stores if s not in train_stores][:k]
        relevant = {s for s in test_new_stores if s in store_set}
        if not relevant:
            continue

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
    embedding_matrix: np.ndarray,
    meta,
    conn: sqlite3.Connection,
    k: int = 5,
    split_date: str = SPLIT_DATE,
) -> dict:
    """Run all model evaluations using temporal train/test split.

    Returns: {
        "k": k,
        "split_date": split_date,
        "n_test_cases": ...,
        "models": [
            {"name": "...", "level": "...", "metrics": {...}},
            ...
        ]
    }
    """
    print(f"  Building temporal test cases (split: {split_date})...", flush=True)
    test_cases = build_temporal_test_cases(conn, split_date)
    print(f"  {len(test_cases)} test cases (customers with new stores after split)", flush=True)

    # Build TRAIN-only matrices
    print("  Building train-only matrices...", flush=True)
    sp_matrix, sp_store_ids, sp_product_ids = build_train_store_product_matrix(conn, split_date)
    cs_matrix, cs_customer_ids, cs_store_ids = build_train_customer_store_matrix(conn, split_date)
    interaction_matrix, mb_customer_ids, mb_store_ids = build_train_interaction_matrix(conn, split_date)
    print(f"  Train SP: {sp_matrix.shape}, CS: {cs_matrix.shape}, "
          f"Interaction: {interaction_matrix.shape}", flush=True)

    # Train model-based CF on train data only
    print("  Training ALS on train data...", flush=True)
    from pipelines.collab_model import (
        train_als, train_lightfm, _build_item_features, _build_user_features,
    )
    als_model = train_als(interaction_matrix)

    print("  Training LightFM WARP on train data...", flush=True)
    lightfm_model = train_lightfm(interaction_matrix)

    print("  Building item + user features for hybrid model...", flush=True)
    item_features = _build_item_features(mb_store_ids, conn)
    user_features = _build_user_features(mb_customer_ids, conn, before_date=split_date)

    print("  Training LightFM Hybrid on train data (128 components, tuned)...", flush=True)
    lightfm_hybrid_model = train_lightfm(
        interaction_matrix,
        item_features=item_features,
        user_features=user_features,
        no_components=128,
        learning_rate=0.01,
        epochs=50,
    )

    models = []

    # 1. Content-Based (CLIP) — no temporal leakage concern
    print("  Evaluating Content-Based (CLIP)...", flush=True)
    cb_metrics = evaluate_content_based(embedding_matrix, meta, k)
    models.append({
        "name": "Content-Based (CLIP)",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in cb_metrics.items()},
    })

    # 2. Item-Based CF (memory)
    print("  Evaluating Item-Based CF...", flush=True)
    ib_metrics = evaluate_item_based_cf(sp_matrix, sp_store_ids, test_cases, k)
    models.append({
        "name": "Item-Based CF",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in ib_metrics.items()},
    })

    # 3. User-Based CF (memory)
    print("  Evaluating User-Based CF...", flush=True)
    ub_metrics = evaluate_user_based_cf(cs_matrix, cs_customer_ids, cs_store_ids, test_cases, k)
    models.append({
        "name": "User-Based CF",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in ub_metrics.items()},
    })

    # 4. ALS
    print("  Evaluating ALS...", flush=True)
    als_metrics = evaluate_als(
        als_model, interaction_matrix, mb_customer_ids, mb_store_ids, test_cases, k,
    )
    models.append({
        "name": "ALS",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in als_metrics.items()},
    })

    # 5. LightFM WARP
    print("  Evaluating LightFM WARP...", flush=True)
    lfm_metrics = evaluate_lightfm(
        lightfm_model, interaction_matrix, mb_customer_ids, mb_store_ids, test_cases, k,
    )
    models.append({
        "name": "LightFM WARP",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in lfm_metrics.items()},
    })

    # 6. LightFM Hybrid
    print("  Evaluating LightFM Hybrid...", flush=True)
    lfm_hybrid_metrics = evaluate_lightfm(
        lightfm_hybrid_model, interaction_matrix, mb_customer_ids, mb_store_ids, test_cases, k,
        item_features=item_features, user_features=user_features,
    )
    models.append({
        "name": "LightFM Hybrid",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in lfm_hybrid_metrics.items()},
    })

    # 7. Random Baseline (store-level only — fair comparison with CF models)
    print("  Evaluating Random Baseline...", flush=True)
    rand_store = evaluate_random_baseline_stores(sp_store_ids, test_cases, k)
    models.append({
        "name": "Random Baseline",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in rand_store.items()},
    })

    # 8. Popularity Baseline (store-level only — fair comparison with CF models)
    print("  Evaluating Popularity Baseline...", flush=True)
    pop_store = evaluate_popularity_baseline_stores(conn, sp_store_ids, test_cases, k, split_date)
    models.append({
        "name": "Popularity Baseline",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in pop_store.items()},
    })

    return {
        "k": k,
        "split_date": split_date,
        "n_test_cases": len(test_cases),
        "models": models,
    }
