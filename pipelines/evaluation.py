"""Offline evaluation of recommendation models.

Uses a temporal train/test split to avoid data leakage:
- Train: all transactions before the split date
- Test: customers who visited NEW stores after the split date
- Models are trained only on train data, then evaluated on test

Models evaluated with six IR metrics (+ novelty for product-level):
- Content-Based (CLIP): self-retrieval, relevance = same category   [Product]
- Item-Based CF (memory): temporal split on store level             [Store]
- User-Based CF (memory): temporal split on store level             [Store]
- ALS (model-based): implicit feedback matrix factorization         [Store]
- LightFM WARP: pairwise ranking loss, no side features             [Store]
- LightFM Hybrid: WARP + store profile + user behaviour features    [Store]
- Random Baseline (Store): uniform random store sample              [Store]
- Popularity Baseline (Store): most-visited stores                  [Store]
- Random Baseline (Product): uniform random product sample          [Product]
- Popularity Baseline (Product): most-common merchant's products    [Product]

Relevance criterion for product-level evaluation:
  - Twinset: same URL-slug category (100% coverage from path segments)
  - Arcaplanet: same animal×product-type category (≈67% coverage from name keywords)
  - Fallback (uncategorized products): same merchant
  This is stricter and more valid than "same merchant" as it avoids inflating
  recall for merchants with large, diverse catalogs (e.g. 499 Arcaplanet products).

Novelty metric (product-level only):
  novelty = mean(-log2(freq(i) / N_queries)) over recommended items, where
  freq(i) is how many queries caused item i to appear in the top-k list.
  High novelty = model surfaces rare/long-tail items rather than always-popular ones.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
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


def novelty_at_k(recommended: list, item_freq: dict, n_queries: int) -> float:
    """Mean self-information of recommended items.

    novelty(i) = -log2(freq(i) / n_queries)

    where freq(i) is the number of queries for which item i appears in the
    top-k list.  Items that appear in every list (freq → n_queries) have
    novelty → 0 (trivially popular); items rarely surfaced have high novelty.
    The list-level score is the mean over all recommended items.

    References: Vargas & Castells, RecSys 2011, "Rank and relevance in novelty
    and diversity metrics for recommender systems."
    """
    if not recommended or n_queries == 0:
        return 0.0
    scores = []
    for item in recommended:
        freq = max(item_freq.get(item, 1), 1)  # floor at 1 to avoid log(0)
        scores.append(-np.log2(freq / n_queries))
    return float(np.mean(scores))


# ── Product Category Extraction ────────────────────────────────────────

_ARCAPLANET_ANIMAL_KEYWORDS = {
    "cat": [
        # Spanish
        "gato", "gata", "felino", "felina", "gatito", "gatita",
        # Italian
        "gatto", "gatta", "gatti", "gattino", "gattini", "felini",
        # English
        "cat", "kitty", "kitten", "feline",
    ],
    "dog": [
        # Spanish
        "perro", "perra", "canino", "canina", "cachorro",
        # Italian
        "cane", "cani", "canino", "cucciolo",
        # English
        "dog", "puppy", "canine",
    ],
    "small_animal": [
        # Spanish
        "conejo", "roedor", "conejillo", "hurón", "pájaro", "loro",
        "pez", "acuario", "tortuga",
        # Italian
        "roditori", "roditore", "criceto", "coniglio", "conigli",
        "pappagallo", "uccello", "uccelli", "pesce", "pesci", "acquario",
        "tartaruga", "furretto", "chinchilla",
        # English
        "rabbit", "hamster", "guinea", "ferret", "bird", "fish", "turtle",
    ],
}
_ARCAPLANET_TYPE_KEYWORDS = {
    "snack": [
        "snack", "premio", "premi", "golosina", "treat",
        "osso", "biscotto", "biscotti", "masticabile", "masticabili",
    ],
    "food": [
        # Spanish
        "pienso", "alimento", "comida", "croqueta", "ración", "paté", "pate",
        # Italian
        "mangime", "cibo", "alimenti", "holistic", "maintenance",
        "acana", "orijen", "wellness", "advance",
        # Generic
        "wet food", "dry food", "pouch", "lata", "fiocchi",
        # Aquarium food
        "flake", "gammarus", "artemia", "spirulina",
    ],
    "litter": [
        "arena", "litter", "arenero", "bandeja", "lettiera",
        "sabbia", "sabbiolina", "toilette",
    ],
    "accessories": [
        # Housing/beds
        "cama", "caseta", "cuccia", "casetta", "castello", "recinto",
        "gabbia", "griglia", "coperchio", "casa", "nido",
        # Bowls/feeders
        "ciotola", "mangiatoia", "fontana", "cucchiaio",
        # Wearable
        "collar", "arnés", "arnes", "correa", "leash", "harness",
        "manta", "coperta", "ropa", "traje",
        # Aquarium equipment
        "tank", "pompa", "filtro", "riscaldatore", "pulitore",
        "cartuccia", "fascia", "luce", "led", "blinki",
        # Other
        "campanello", "trasportino", "cuscinetto",
    ],
    "toy": [
        # Spanish
        "juguete", "pelota", "rascador",
        # Italian
        "gioco", "giochi", "corda", "canne", "canna", "piuma", "piume",
        "aquilone", "pallina", "tunnel",
        # English
        "toy", "ball", "feather",
    ],
    "grooming": [
        "champú", "champu", "shampoo", "igiene", "higiene",
        "cepillo", "spazzola", "brush", "perfume",
        "toallita", "wipe",
    ],
    "health": [
        # Spanish
        "antiparasit", "pipeta", "vacuna", "suplemento", "vitamina",
        # Italian
        "antiparassitario", "compresse", "spray", "spot on",
        "advantix", "adaptil", "frontline",
        # English
        "supplement", "vitamin",
    ],
}


def _extract_twinset_category(url: str) -> str | None:
    """Extract broad category from Twinset product URL slug.

    Twinset URL structure: .../es-es/{category-words}-{product-code}.html
    Product codes follow the pattern: 3-digits + 2-uppercase + alphanum + _ + 5-digits
    e.g. 261TP3083_00059, 261LB3GAA_00006, 261AP2254_10780

    After stripping the code, we take the FIRST word of the remaining slug as the
    broad product category (vestido, falda, jersey, zapatos, etc.).  This gives
    ~12 distinct categories suitable for relevance grouping.

    Common prefixes (maxi, mini, micro, over) are merged into the base word.
    """
    if not url:
        return None
    path = url.split("?")[0].rstrip("/")
    seg = path.split("/")[-1].replace(".html", "")

    # Strip trailing product code: -\d{3}[A-Z]{2}[A-Z0-9]*_\d+ pattern
    clean = re.sub(r"-\d{3}[A-Z]{2}[A-Z0-9]*_\d+$", "", seg).lower()
    if clean == seg.lower() or not clean:
        return None  # code pattern not found — URL format unexpected

    # Take the first hyphen-separated word as the broad category
    words = clean.split("-")
    first_word = words[0] if words else None
    if not first_word:
        return None

    # Merge size prefixes into next word (maxi+jersey → jersey, mini+vestido → vestido)
    size_prefixes = {"maxi", "mini", "micro", "over", "midi"}
    if first_word in size_prefixes and len(words) > 1:
        first_word = words[1]

    # Normalise common variants to canonical forms
    _norm = {
        "pantalones": "pantalon", "vestidos": "vestido", "jerseis": "jersey",
        "chaquetas": "chaqueta", "faldas": "falda", "blusas": "blusa",
        "camisas": "camisa", "abrigos": "abrigo", "zapatos": "zapato",
        "sandalias": "sandalia", "bolsos": "bolso", "sneakers": "zapato",
        "mocasines": "zapato", "botas": "zapato", "stiletto": "zapato",
        "leggings": "pantalon", "vaqueros": "pantalon", "bermudas": "pantalon",
        "shorts": "pantalon", "blazer": "chaqueta", "cardigan": "jersey",
        "sweater": "jersey", "bodies": "body", "jumpsuit": "vestido",
        "mono": "vestido", "overall": "vestido",
        # Maxi/mini prefixes that weren't separated by a hyphen in the URL
        "maxijersey": "jersey", "maxivestido": "vestido", "maxifalda": "falda",
        "minivestido": "vestido", "minifalda": "falda",
        "camiseta": "camisa", "camisetas": "camisa",
    }
    return _norm.get(first_word, first_word)


def _extract_arcaplanet_category(name: str) -> str | None:
    """Extract animal×product-type category from Arcaplanet product name."""
    name_lower = name.lower()
    animal = None
    for key, keywords in _ARCAPLANET_ANIMAL_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            animal = key
            break
    ptype = None
    for key, keywords in _ARCAPLANET_TYPE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            ptype = key
            break
    if animal and ptype:
        return f"{animal}_{ptype}"
    elif animal:
        return f"{animal}_other"
    elif ptype:
        return f"other_{ptype}"
    return None


def extract_product_categories(meta: pd.DataFrame) -> pd.Series:
    """Extract product categories for relevance assessment.

    Returns a Series of category strings aligned with meta index.
    None means uncategorized — falls back to same-merchant relevance.

    Coverage (approximate):
    - Twinset: ~100% via URL slug extraction
    - Arcaplanet: ~67% via name keyword matching (163/500 are "other_other" → None)
    """
    categories = []
    for _, row in meta.iterrows():
        merchant = str(row.get("merchant", "")).lower()
        name = str(row.get("name", ""))
        url = str(row.get("url", ""))
        if "twinset" in merchant:
            cat = _extract_twinset_category(url)
        elif "arcaplanet" in merchant:
            cat = _extract_arcaplanet_category(name)
        else:
            cat = None
        categories.append(cat)
    return pd.Series(categories, index=meta.index)


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
    """Build customer-store matrix from TRAIN data only.

    Values = distinct invoice (visit) count per customer-store pair, consistent
    with the interaction matrix used by model-based CF.  The previous version
    used SUM(qty * unit_price) as the cell value, which biased cosine similarity
    toward high-spending customers at expensive stores — conflating price level
    with preference strength.  Visit counts are a more neutral signal that
    reflects actual behavioural frequency regardless of merchant price point.
    """
    rows = conn.execute(
        "SELECT customer_id, store_id, COUNT(DISTINCT invoice_id) as visits "
        "FROM transactions WHERE issued_on < ? "
        "GROUP BY customer_id, store_id",
        (split_date,),
    ).fetchall()

    customer_set = sorted(set(r[0] for r in rows))
    store_set = sorted(set(r[1] for r in rows))
    customer_idx = {c: i for i, c in enumerate(customer_set)}
    store_idx = {s: i for i, s in enumerate(store_set)}

    row_indices, col_indices, values = [], [], []
    for cid, sid, visits in rows:
        row_indices.append(customer_idx[cid])
        col_indices.append(store_idx[sid])
        values.append(float(visits))

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


def _build_relevant_sets(meta: pd.DataFrame) -> list[set]:
    """Build per-product relevant sets using category-based relevance.

    Relevance criterion (in priority order):
    1. Same merchant AND same extracted category (Twinset / Arcaplanet)
    2. Fallback: same merchant only (for uncategorized products)

    This is stricter than same-merchant alone and avoids inflating recall for
    merchants with heterogeneous catalogs (e.g. 499 Arcaplanet products span
    food, grooming, toys — treating them all as equally relevant is not valid).
    """
    n = len(meta)
    merchants = meta["merchant"].values
    categories = extract_product_categories(meta).values  # may contain None

    # Pre-group indices by merchant and by (merchant, category)
    merchant_groups: dict[str, list[int]] = defaultdict(list)
    cat_groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        merchant_groups[merchants[i]].append(i)
        if categories[i] is not None:
            cat_groups[(merchants[i], categories[i])].append(i)

    relevant_sets = []
    for i in range(n):
        m = merchants[i]
        c = categories[i]
        if c is not None:
            # Use category-based relevance
            relevant = {j for j in cat_groups[(m, c)] if j != i}
        else:
            # Fallback: same merchant
            relevant = {j for j in merchant_groups[m] if j != i}
        relevant_sets.append(relevant)

    return relevant_sets


def evaluate_content_based(matrix: np.ndarray, meta, k: int) -> dict:
    """Evaluate content-based CLIP model via self-retrieval.

    For each product, use its embedding as query, exclude self,
    and check if products from the same category appear in top-k.

    Relevance = same category (from URL slugs for Twinset, name keywords for
    Arcaplanet).  Uncategorized products fall back to same-merchant relevance.
    See _build_relevant_sets() for the full logic.

    Beyond accuracy: ILD diversity (embedding cosine dissimilarity) and
    novelty (mean self-information based on recommendation frequency).
    """
    n = matrix.shape[0]
    all_recs = []
    metrics_list = []

    # Pre-build relevant sets with category-based relevance
    relevant_sets = _build_relevant_sets(meta)

    # First pass: collect all recommendation lists (needed for novelty freq)
    raw_recs = []
    for i in range(n):
        query = matrix[i : i + 1]
        scores = (matrix @ query.T).flatten()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[::-1][:k]
        raw_recs.append(top_indices.tolist())

    # Build recommendation frequency map for novelty
    item_freq: dict[int, int] = Counter(item for recs in raw_recs for item in recs)
    n_queries = n  # one query per product

    for i in range(n):
        recommended = raw_recs[i]
        relevant = relevant_sets[i]
        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    # Diversity: intra-list embedding cosine dissimilarity
    def sim_fn(a, b):
        return float(matrix[a] @ matrix[b])

    div_scores = [diversity(recs, sim_fn) for recs in all_recs if len(recs) >= 2]

    # Novelty: mean self-information over recommended items
    nov_scores = [novelty_at_k(recs, item_freq, n_queries) for recs in all_recs]

    # Log category coverage for transparency
    cats = extract_product_categories(meta)
    n_categorized = cats.notna().sum()

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": np.mean(div_scores) if div_scores else 0.0,
        "novelty": np.mean(nov_scores) if nov_scores else 0.0,
        "n_categorized": int(n_categorized),
        "n_products": n,
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
    """Random baseline for product-level: uniform random sample of k products.

    Uses the same category-based relevance criterion as evaluate_content_based()
    for a fair apples-to-apples comparison.
    """
    rng = np.random.RandomState(42)
    n = matrix.shape[0]
    all_recs = []
    metrics_list = []

    relevant_sets = _build_relevant_sets(meta)

    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        recommended = rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()
        m = compute_all_metrics(recommended, relevant_sets[i], k)
        metrics_list.append(m)
        all_recs.append(recommended)

    item_freq: dict[int, int] = Counter(item for recs in all_recs for item in recs)
    nov_scores = [novelty_at_k(recs, item_freq, n) for recs in all_recs]

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": 1.0,
        "novelty": np.mean(nov_scores) if nov_scores else 0.0,
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
    """Popularity baseline: always recommend the most common merchant's products.

    Uses the same category-based relevance criterion as evaluate_content_based()
    for a fair apples-to-apples comparison.  Novelty will be very low because
    the same small set of products is recommended for every query.
    """
    n = matrix.shape[0]
    merchants = meta["merchant"].values

    merchant_counts = Counter(merchants)
    most_popular_merchant = merchant_counts.most_common(1)[0][0]
    popular_indices = [i for i in range(n) if merchants[i] == most_popular_merchant]

    all_recs = []
    metrics_list = []

    relevant_sets = _build_relevant_sets(meta)

    for i in range(n):
        candidates = [j for j in popular_indices if j != i][:k]
        m = compute_all_metrics(candidates, relevant_sets[i], k)
        metrics_list.append(m)
        all_recs.append(candidates)

    item_freq: dict[int, int] = Counter(item for recs in all_recs for item in recs)
    nov_scores = [novelty_at_k(recs, item_freq, n) for recs in all_recs]

    return {
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "ndcg": np.mean([m["ndcg"] for m in metrics_list]),
        "hit_rate": np.mean([m["hit_rate"] for m in metrics_list]),
        "coverage": coverage(all_recs, n),
        "diversity": 0.0,
        "novelty": np.mean(nov_scores) if nov_scores else 0.0,
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


# ── Demographic Popularity Evaluator ─────────────────────────────────────


def evaluate_demographic_popularity(
    test_cases: list,
    store_ids: list[str],
    demo_dict: dict,
    conn: sqlite3.Connection,
    k: int,
    split_date: str = SPLIT_DATE,
) -> dict:
    """Evaluate segment-based Demographic Popularity recommender.

    For each test user, recommends the most-visited stores (by visit count)
    within their demographic segment (nationality × age_bin × tourist_type)
    using TRAIN-period transactions only.

    Fallback hierarchy when a segment has fewer than MIN_SEGMENT_SIZE
    distinct customers:
        full segment (nat + age + tourist)
        → nationality + age_bin
        → nationality only
        → global popularity

    This is a cold-start model: it does not use the test user's own
    transaction history to generate scores, only their demographic profile.
    The test user's train_stores are still excluded from recommendations
    to avoid trivial matches.
    """
    from pipelines.demographic import MIN_SEGMENT_SIZE

    store_set = set(store_ids)

    # Build segment → store visit counts from training data only
    rows = conn.execute(
        "SELECT t.customer_id, t.store_id, COUNT(DISTINCT t.invoice_id) as visits "
        "FROM transactions t "
        "WHERE t.issued_on < ? "
        "GROUP BY t.customer_id, t.store_id",
        (split_date,),
    ).fetchall()

    seg_full: dict = defaultdict(lambda: defaultdict(int))
    seg_nat_age: dict = defaultdict(lambda: defaultdict(int))
    seg_nat: dict = defaultdict(lambda: defaultdict(int))
    global_counts: dict = defaultdict(int)
    seg_full_n: dict = defaultdict(set)
    seg_nat_age_n: dict = defaultdict(set)
    seg_nat_n: dict = defaultdict(set)

    for cid, sid, visits in rows:
        if sid not in store_set:
            continue
        d = demo_dict.get(cid, {})
        cn = d.get("nationality", "nat:unknown")
        ca = d.get("age_bin", "age:unknown")
        ct = d.get("tourist_type", "tourist:international")

        seg_full[(cn, ca, ct)][sid] += visits
        seg_full_n[(cn, ca, ct)].add(cid)
        seg_nat_age[(cn, ca)][sid] += visits
        seg_nat_age_n[(cn, ca)].add(cid)
        seg_nat[cn][sid] += visits
        seg_nat_n[cn].add(cid)
        global_counts[sid] += visits

    def rank_stores(counts: dict, exclude: set) -> list[str]:
        return [
            s for s, _ in sorted(counts.items(), key=lambda x: -x[1])
            if s in store_set and s not in exclude
        ]

    global_ranked = rank_stores(global_counts, set())

    all_recs = []
    metrics_list = []

    for cid, train_stores, test_new_stores in test_cases:
        d = demo_dict.get(cid, {})
        nat = d.get("nationality", "nat:unknown")
        age = d.get("age_bin", "age:unknown")
        tourist = d.get("tourist_type", "tourist:international")
        full_seg = (nat, age, tourist)
        nat_age_seg = (nat, age)

        relevant = {s for s in test_new_stores if s in store_set}
        if not relevant:
            continue

        if len(seg_full_n[full_seg]) >= MIN_SEGMENT_SIZE:
            ranked = rank_stores(seg_full[full_seg], train_stores)
        elif len(seg_nat_age_n[nat_age_seg]) >= MIN_SEGMENT_SIZE:
            ranked = rank_stores(seg_nat_age[nat_age_seg], train_stores)
        elif len(seg_nat_n[nat]) >= MIN_SEGMENT_SIZE:
            ranked = rank_stores(seg_nat[nat], train_stores)
        else:
            ranked = [s for s in global_ranked if s not in train_stores]

        recommended = ranked[:k]
        m = compute_all_metrics(recommended, relevant, k)
        metrics_list.append(m)
        all_recs.append(recommended)

    if not metrics_list:
        return {m: 0.0 for m in ["precision", "recall", "ndcg", "hit_rate", "coverage", "diversity"]}

    return {
        "precision": float(np.mean([m["precision"] for m in metrics_list])),
        "recall": float(np.mean([m["recall"] for m in metrics_list])),
        "ndcg": float(np.mean([m["ndcg"] for m in metrics_list])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in metrics_list])),
        "coverage": coverage(all_recs, len(store_ids)),
        "diversity": 0.0,  # popularity-based models produce no embedding diversity
    }


# ── Master Evaluation Function ─────────────────────────────────────────


def run_full_evaluation(
    embedding_matrix: np.ndarray,
    meta,
    conn: sqlite3.Connection,
    k: int = 5,
    split_date: str = SPLIT_DATE,
    demo_dict: Optional[dict] = None,
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

    # ── Compute data_stats ──────────────────────────────────────────────
    n_customers_train = cs_matrix.shape[0]
    n_stores_train = len(sp_store_ids)
    n_products = len(sp_product_ids)
    n_embeddings = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]

    n_train_transactions = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE issued_on < ?", (split_date,),
    ).fetchone()[0]
    n_test_transactions = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE issued_on >= ?", (split_date,),
    ).fetchone()[0]

    date_range_row = conn.execute(
        "SELECT MIN(issued_on), MAX(issued_on) FROM transactions",
    ).fetchone()
    date_range_min = date_range_row[0] if date_range_row else None
    date_range_max = date_range_row[1] if date_range_row else None

    # Sparsity = 1 - (nnz / total_cells)
    cs_total = cs_matrix.shape[0] * cs_matrix.shape[1]
    sparsity_customer_store = round(1.0 - (cs_matrix.nnz / cs_total), 6) if cs_total > 0 else 1.0
    sp_total = sp_matrix.shape[0] * sp_matrix.shape[1]
    sparsity_store_product = round(1.0 - (sp_matrix.nnz / sp_total), 6) if sp_total > 0 else 1.0

    # Average stores per customer (from CS matrix)
    stores_per_customer = np.diff(cs_matrix.indptr)  # nnz per row
    avg_stores_per_customer = round(float(stores_per_customer.mean()), 2) if len(stores_per_customer) > 0 else 0.0

    # Average products per store (from SP matrix)
    products_per_store = np.diff(sp_matrix.indptr)  # nnz per row
    avg_products_per_store = round(float(products_per_store.mean()), 2) if len(products_per_store) > 0 else 0.0

    data_stats = {
        "n_customers_train": n_customers_train,
        "n_stores_train": n_stores_train,
        "n_products": n_products,
        "n_train_transactions": n_train_transactions,
        "n_test_transactions": n_test_transactions,
        "date_range_min": date_range_min,
        "date_range_max": date_range_max,
        "split_date": split_date,
        "n_test_cases": len(test_cases),
        "sparsity_customer_store": sparsity_customer_store,
        "sparsity_store_product": sparsity_store_product,
        "avg_stores_per_customer": avg_stores_per_customer,
        "avg_products_per_store": avg_products_per_store,
        "n_embeddings": n_embeddings,
        "embedding_dim": embedding_dim,
    }

    models = []

    # 1. Content-Based (CLIP) — no temporal leakage concern
    print("  Evaluating Content-Based (CLIP)...", flush=True)
    cb_metrics = evaluate_content_based(embedding_matrix, meta, k)
    # Pull out non-IR statistics (n_categorized, n_products) into a separate field
    _cb_stats_keys = {"n_categorized", "n_products"}
    cb_stats = {k: int(cb_metrics[k]) for k in _cb_stats_keys if k in cb_metrics}
    cb_ir = {k: v for k, v in cb_metrics.items() if k not in _cb_stats_keys}
    print(f"    Category coverage: {cb_stats.get('n_categorized', '?')}"
          f"/{cb_stats.get('n_products', '?')} products", flush=True)
    models.append({
        "name": "Content-Based (CLIP)",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in cb_ir.items()},
        "stats": cb_stats,
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

    # 7. Random Baseline — Store level (CF comparison)
    print("  Evaluating Random Baseline (Store)...", flush=True)
    rand_store = evaluate_random_baseline_stores(sp_store_ids, test_cases, k)
    models.append({
        "name": "Random Baseline",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in rand_store.items()},
    })

    # 8. Popularity Baseline — Store level (CF comparison)
    print("  Evaluating Popularity Baseline (Store)...", flush=True)
    pop_store = evaluate_popularity_baseline_stores(conn, sp_store_ids, test_cases, k, split_date)
    models.append({
        "name": "Popularity Baseline",
        "level": "Store",
        "metrics": {key: round(float(val), 4) for key, val in pop_store.items()},
    })

    # 9. Random Baseline — Product level (content-based comparison)
    # Uses the same self-retrieval protocol as CLIP: relevance = same merchant.
    # Answers the question: does CLIP actually outperform a naive product-level
    # baseline, or does the evaluation protocol inflate its apparent accuracy?
    print("  Evaluating Random Baseline (Product)...", flush=True)
    rand_product = evaluate_random_baseline_products(embedding_matrix, meta, k)
    models.append({
        "name": "Random Baseline (Product)",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in rand_product.items()},
    })

    # 10. Popularity Baseline — Product level (content-based comparison)
    print("  Evaluating Popularity Baseline (Product)...", flush=True)
    pop_product = evaluate_popularity_baseline_products(embedding_matrix, meta, k)
    models.append({
        "name": "Popularity Baseline (Product)",
        "level": "Product",
        "metrics": {key: round(float(val), 4) for key, val in pop_product.items()},
    })

    # ── Demographic models (optional — only when demo_dict provided) ────────
    if demo_dict is not None:
        from pipelines.demographic import (
            _build_demographic_user_features,
            _build_full_hybrid_user_features,
        )

        # 11. Demographic Popularity — segment-based cold-start baseline
        print("  Evaluating Demographic Popularity...", flush=True)
        demo_pop_metrics = evaluate_demographic_popularity(
            test_cases, mb_store_ids, demo_dict, conn, k, split_date,
        )
        models.append({
            "name": "Demographic Popularity",
            "level": "Store",
            "metrics": {key: round(float(val), 4) for key, val in demo_pop_metrics.items()},
        })

        # 12. LightFM Demo — demographic user features + store item features.
        # Trained on the same interaction matrix as LightFM WARP but user
        # representation is demographic-only.  Directly comparable to
        # LightFM Hybrid (same item features, different user features).
        print("  Building demographic user features for LightFM Demo...", flush=True)
        demo_user_features = _build_demographic_user_features(mb_customer_ids, demo_dict)
        print("  Training LightFM Demo (64 components, demographic features)...", flush=True)
        lightfm_demo_model = train_lightfm(
            interaction_matrix,
            item_features=item_features,
            user_features=demo_user_features,
            no_components=64,
            learning_rate=0.05,
            epochs=30,
        )
        print("  Evaluating LightFM Demo...", flush=True)
        lfm_demo_metrics = evaluate_lightfm(
            lightfm_demo_model, interaction_matrix, mb_customer_ids, mb_store_ids,
            test_cases, k,
            item_features=item_features,
            user_features=demo_user_features,
        )
        models.append({
            "name": "LightFM Demo",
            "level": "Store",
            "metrics": {key: round(float(val), 4) for key, val in lfm_demo_metrics.items()},
        })

        # 13. LightFM Full Hybrid — behavioral + demographic user features +
        # store item features.  Tests whether adding demographic signals on top
        # of the existing behavioral hybrid improves recommendations.
        print("  Building full hybrid user features (behavioral + demographic)...", flush=True)
        full_hybrid_user_features = _build_full_hybrid_user_features(
            mb_customer_ids, conn, demo_dict, before_date=split_date,
        )
        print("  Training LightFM Full Hybrid (128 components, all features)...", flush=True)
        lightfm_full_hybrid_model = train_lightfm(
            interaction_matrix,
            item_features=item_features,
            user_features=full_hybrid_user_features,
            no_components=128,
            learning_rate=0.01,
            epochs=50,
        )
        print("  Evaluating LightFM Full Hybrid...", flush=True)
        lfm_full_hybrid_metrics = evaluate_lightfm(
            lightfm_full_hybrid_model, interaction_matrix, mb_customer_ids, mb_store_ids,
            test_cases, k,
            item_features=item_features,
            user_features=full_hybrid_user_features,
        )
        models.append({
            "name": "LightFM Full Hybrid",
            "level": "Store",
            "metrics": {key: round(float(val), 4) for key, val in lfm_full_hybrid_metrics.items()},
        })

    return {
        "k": k,
        "split_date": split_date,
        "n_test_cases": len(test_cases),
        "data_stats": data_stats,
        "models": models,
    }
