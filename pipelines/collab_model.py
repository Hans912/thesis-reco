"""Model-based collaborative filtering using ALS and LightFM.

Three model variants:
1. ALS (implicit library) — weighted matrix factorization for implicit feedback
2. LightFM WARP — pairwise ranking loss, no side features
3. LightFM WARP + features — hybrid model with store profile features

All models operate on the customer-store interaction matrix.
Memory-based CF in collab.py is preserved and unchanged.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ── Helpers ─────────────────────────────────────────────────────────────


def _build_interaction_matrix(
    conn: sqlite3.Connection,
    before_date: Optional[str] = None,
    recency_decay: float = 180.0,
):
    """Build a sparse customer x store matrix from transactions.

    Values = recency-weighted visit count. Each visit is weighted by
    exp(-days_since_visit / recency_decay), so recent visits count more.
    A 180-day half-life means a visit from 6 months ago has ~37% weight.

    Args:
        before_date: Only use transactions before this date (for temporal split).
        recency_decay: Decay constant in days. Higher = slower decay.

    Returns (matrix, customer_ids, store_ids).
    """
    date_filter = "WHERE issued_on < ?" if before_date else ""
    params = (before_date,) if before_date else ()

    # Get the reference date (latest transaction or split date)
    ref_date = before_date or conn.execute(
        "SELECT MAX(issued_on) FROM transactions"
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT customer_id, store_id, "
        f"  COUNT(DISTINCT invoice_id) as visits, "
        f"  julianday(?) - julianday(MAX(issued_on)) as days_ago "
        f"FROM transactions {date_filter} "
        f"GROUP BY customer_id, store_id",
        (ref_date,) + params,
    ).fetchall()

    customer_set = sorted(set(r[0] for r in rows))
    store_set = sorted(set(r[1] for r in rows))
    customer_idx = {c: i for i, c in enumerate(customer_set)}
    store_idx = {s: i for i, s in enumerate(store_set)}

    row_indices = []
    col_indices = []
    values = []
    for customer_id, store_id, visits, days_ago in rows:
        # Recency-weighted: visits * exp(-days_ago / decay)
        days = max(float(days_ago or 0), 0)
        weight = float(visits) * np.exp(-days / recency_decay)
        row_indices.append(customer_idx[customer_id])
        col_indices.append(store_idx[store_id])
        values.append(weight)

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(customer_set), len(store_set)),
    )
    return matrix, customer_set, store_set


def _city_to_region(city: str) -> str:
    """Map a store city to a geographic region for LightFM item features.

    Raw city names have two problems: inconsistent capitalisation across the
    dataset (COMO / Como / como all appear) and high cardinality (~198 distinct
    values after normalisation).  Cities with only 1-2 stores yield essentially
    untrained embeddings that add noise rather than signal.

    Mapping to 6 regions gives every feature bucket enough stores to learn a
    meaningful embedding while still capturing the geographic patterns that
    matter for this cross-border tourist dataset:
      - lake_como_zone: the primary tax-free shopping hub near the Swiss border
      - milan_metro: Italy's main fashion/retail centre
      - alpine_north: other Northern Italian cities and mountain resorts
      - northeast_italy: Veneto / Friuli / Trentino
      - central_italy: Rome, Tuscany, Emilia-Romagna
      - south_italy: Naples and south

    Cities not in any explicit list fall back to their normalized name only if
    they appear frequently enough (>=10 stores); otherwise mapped to 'other'.
    This preserves signal for the handful of major cities while collapsing the
    long tail into a single bucket.
    """
    if not city:
        return "region:other"

    c = city.strip().title()

    LAKE_COMO = {
        "Como", "Cernobbio", "Lavena Ponte Tresa", "Porlezza",
        "Lurate Caccivio", "Cantù", "Mariano Comense", "Bosisio Parini",
        "Meda", "Lecco", "Morbegno", "Bormio", "Colico", "Gravedona",
        "Menaggio", "Tremezzina", "Varenna", "Gazzada Schianno",
        "Cantello", "Olgiate Olona",
    }
    MILAN_METRO = {
        "Milano", "Monza", "Sesto San Giovanni", "Rho", "Cinisello Balsamo",
        "Busto Arsizio", "Legnano", "Cologno Monzese", "Paderno Dugnano",
        "Corsico", "Sesto Calende", "Gallarate",
    }
    ALPINE_NORTH = {
        "Varese", "Luino", "Domodossola", "Verbania", "Aosta", "Torino",
        "Bolzano", "Merano", "Trento", "Brescia", "Bergamo",
        "Cuneo", "Novara", "Vercelli", "Biella", "Alessandria",
    }
    NORTHEAST = {
        "Vicenza", "Verona", "Venezia", "Padova", "Treviso", "Udine",
        "Trieste", "Pordenone", "Belluno", "Rovigo", "Ferrara",
    }
    CENTRAL = {
        "Roma", "Firenze", "Bologna", "Carpi", "Modena", "Perugia",
        "Ancona", "Pisa", "Siena", "Livorno", "Arezzo", "Reggio Emilia",
        "Parma", "Piacenza", "Ravenna", "Rimini", "Pesaro",
    }
    SOUTH = {
        "Napoli", "Foggia", "Telese Terme", "Palermo", "Bari",
        "Catanzaro", "Reggio Calabria", "Catania", "Messina",
        "Salerno", "Taranto", "Brindisi", "Lecce", "Potenza",
        "Campobasso", "L'Aquila", "Pescara",
    }

    if c in LAKE_COMO:
        return "region:lake_como"
    if c in MILAN_METRO:
        return "region:milan_metro"
    if c in ALPINE_NORTH:
        return "region:alpine_north"
    if c in NORTHEAST:
        return "region:northeast"
    if c in CENTRAL:
        return "region:central"
    if c in SOUTH:
        return "region:south"
    return "region:other"


def _build_item_features(store_ids: list[str], conn: sqlite3.Connection):
    """Build LightFM item feature matrix from store profiles.

    Features: geographic region, store size bin, price bin.

    City is mapped to one of 6 geographic regions instead of using raw city
    names.  Raw cities have 198 distinct values after normalisation; ~130 of
    those have fewer than 10 stores, giving LightFM too little data to learn
    meaningful embeddings for those feature dimensions.  Region grouping
    ensures every feature bucket is well-populated.

    Merchant name is intentionally excluded — it is too specific and causes
    the model to recommend only same-merchant stores, overriding the
    collaborative signal.

    Size bins use product-count quartiles from the store_profiles data:
      small  < 50 products  (bottom ~50 % of stores)
      medium   50–300        (middle range)
      large  > 300           (top tier)

    Price bins use median_unit_price thresholds anchored to the dataset:
      low    < 20 EUR
      mid    20–100 EUR
      high   > 100 EUR
    """
    from lightfm.data import Dataset

    profiles = {}
    for sid in store_ids:
        row = conn.execute(
            "SELECT city, num_distinct_products, median_unit_price "
            "FROM store_profiles WHERE store_id = ?",
            (sid,),
        ).fetchone()
        if row:
            profiles[sid] = {
                "region": _city_to_region(row[0]),
                "num_products": row[1] or 0,
                "median_price": row[2] or 0,
            }
        else:
            profiles[sid] = {
                "region": "region:other",
                "num_products": 0,
                "median_price": 0,
            }

    all_regions = {p["region"] for p in profiles.values()}
    size_bins = {"size:small", "size:medium", "size:large"}
    price_bins = {"price:low", "price:mid", "price:high"}
    all_features = list(all_regions | size_bins | price_bins)

    dataset = Dataset()
    dataset.fit(
        users=range(1),  # dummy, we only need item features
        items=range(len(store_ids)),
        item_features=all_features,
    )

    item_feature_list = []
    for i, sid in enumerate(store_ids):
        p = profiles[sid]

        if p["num_products"] < 50:
            size_bin = "size:small"
        elif p["num_products"] < 300:
            size_bin = "size:medium"
        else:
            size_bin = "size:large"

        if p["median_price"] < 20:
            price_bin = "price:low"
        elif p["median_price"] < 100:
            price_bin = "price:mid"
        else:
            price_bin = "price:high"

        item_feature_list.append((i, [p["region"], size_bin, price_bin]))

    item_features = dataset.build_item_features(item_feature_list)
    return item_features


def _build_user_features(
    customer_ids: list[str],
    conn: sqlite3.Connection,
    before_date: Optional[str] = None,
):
    """Build LightFM user feature matrix from transaction history.

    Features: visit frequency bin, spend level bin, primary city.
    """
    from lightfm.data import Dataset

    date_filter = "WHERE issued_on < ?" if before_date else ""
    params = (before_date,) if before_date else ()

    profiles = {}
    for cid in customer_ids:
        row = conn.execute(
            f"SELECT COUNT(DISTINCT store_id), SUM(qty * unit_price) "
            f"FROM transactions {date_filter} "
            f"{'AND' if before_date else 'WHERE'} customer_id = ?",
            params + (cid,),
        ).fetchone()

        n_stores = row[0] or 0
        total_spend = row[1] or 0

        # Primary city (most-visited)
        city_row = conn.execute(
            f"SELECT sp.city, COUNT(*) as cnt "
            f"FROM transactions t JOIN store_profiles sp ON t.store_id = sp.store_id "
            f"{'WHERE t.issued_on < ?' if before_date else ''} "
            f"{'AND' if before_date else 'WHERE'} t.customer_id = ? "
            f"GROUP BY sp.city ORDER BY cnt DESC LIMIT 1",
            params + (cid,),
        ).fetchone()
        primary_city = city_row[0] if city_row else "unknown"

        # Frequency bin — boundaries match the actual visit distribution:
        # 78 % single, 17 % casual (2-3), 5 % regular (4-8), 0.5 % power (9+)
        if n_stores <= 1:
            freq = "freq:single"
        elif n_stores <= 3:
            freq = "freq:casual"
        elif n_stores <= 8:
            freq = "freq:regular"
        else:
            freq = "freq:power"

        # Spend bin — thresholds set at empirical P33/P66 of the full customer
        # spend distribution (290 / 890 EUR) to create three roughly equal-sized
        # buckets.  The original round-number thresholds (200 / 1000) were
        # miscalibrated: the mean spend is heavily skewed by outliers, while the
        # median is ~490 EUR, so most customers would fall in spend:low with
        # the old thresholds.
        if total_spend < 290:
            spend = "spend:low"
        elif total_spend < 890:
            spend = "spend:mid"
        else:
            spend = "spend:high"

        profiles[cid] = {
            "freq": freq,
            "spend": spend,
            "city": f"ucity:{primary_city}",
        }

    # Collect all unique feature values
    all_features = set()
    for p in profiles.values():
        all_features.update([p["freq"], p["spend"], p["city"]])
    all_features = sorted(all_features)

    dataset = Dataset()
    dataset.fit(
        users=range(len(customer_ids)),
        items=range(1),  # dummy, we only need user features
        user_features=all_features,
    )

    user_feature_list = []
    for i, cid in enumerate(customer_ids):
        p = profiles[cid]
        user_feature_list.append((i, [p["freq"], p["spend"], p["city"]]))

    user_features = dataset.build_user_features(user_feature_list)
    return user_features


# ── ALS Model ──────────────────────────────────────────────────────────


def train_als(
    interaction_matrix: sparse.csr_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 30,
):
    """Train ALS model for implicit feedback.

    Returns trained model.
    """
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
    )
    # implicit 0.7+ expects user-item matrix (customers × stores)
    model.fit(interaction_matrix)
    return model


def als_recommend(model, user_idx: int, interaction_matrix: sparse.csr_matrix, k: int = 5):
    """Get top-k store recommendations for a user from ALS model."""
    ids, scores = model.recommend(
        user_idx,
        interaction_matrix[user_idx],
        N=k,
        filter_already_liked_items=True,
    )
    return ids.tolist(), scores.tolist()


def als_similar_items(model, item_idx: int, k: int = 5):
    """Get top-k similar stores from ALS model's item factors."""
    ids, scores = model.similar_items(item_idx, N=k + 1)
    # Filter out self
    result_ids = []
    result_scores = []
    for i, s in zip(ids, scores):
        if i != item_idx:
            result_ids.append(int(i))
            result_scores.append(float(s))
    return result_ids[:k], result_scores[:k]


# ── LightFM Model ─────────────────────────────────────────────────────


def train_lightfm(
    interaction_matrix: sparse.csr_matrix,
    item_features=None,
    user_features=None,
    no_components: int = 64,
    learning_rate: float = 0.05,
    epochs: int = 30,
    loss: str = "warp",
):
    """Train LightFM model with WARP loss.

    If item_features/user_features are provided, trains a hybrid model.
    Returns trained model.
    """
    from lightfm import LightFM

    model = LightFM(
        no_components=no_components,
        learning_rate=learning_rate,
        loss=loss,
        random_state=42,
    )
    model.fit(
        interaction_matrix,
        item_features=item_features,
        user_features=user_features,
        epochs=epochs,
        num_threads=4,
    )
    return model


def lightfm_recommend(
    model, user_idx: int, n_items: int,
    interaction_matrix: sparse.csr_matrix,
    item_features=None, user_features=None, k: int = 5,
):
    """Get top-k store recommendations for a user from LightFM model."""
    scores = model.predict(
        user_idx, np.arange(n_items),
        item_features=item_features, user_features=user_features,
    )

    # Filter out already-visited stores
    visited = set(interaction_matrix[user_idx].nonzero()[1])
    for v in visited:
        scores[v] = -np.inf

    top_indices = np.argsort(-scores)[:k]
    return top_indices.tolist(), scores[top_indices].tolist()


def lightfm_similar_items(model, item_idx: int, item_features=None, k: int = 5):
    """Get top-k similar stores from LightFM item embeddings."""
    _, item_embeddings = model.get_item_representations(features=item_features)
    query = item_embeddings[item_idx : item_idx + 1]
    sims = sklearn_cosine(query, item_embeddings).flatten()
    sims[item_idx] = -1  # exclude self
    top_indices = np.argsort(-sims)[:k]
    return top_indices.tolist(), sims[top_indices].tolist()


# ── High-Level Recommendation with Dedup ──────────────────────────────


def recommend_stores_lightfm(
    customer_id: str,
    model_data: dict,
    conn: sqlite3.Connection,
    top_k: int = 5,
) -> list[dict]:
    """Recommend stores for a customer using LightFM WARP model.

    Handles customer lookup, score normalization, merchant+city dedup,
    and store profile enrichment. Returns list of dicts matching the
    StoreRecommendation format used by other CF endpoints.
    """
    customer_ids = model_data["customer_ids"]
    store_ids = model_data["store_ids"]
    model = model_data["lightfm_model"]
    interaction_matrix = model_data["interaction_matrix"]

    if customer_id not in customer_ids:
        return []

    cidx = customer_ids.index(customer_id)
    n_items = len(store_ids)

    scores = model.predict(cidx, np.arange(n_items))

    # Filter out already-visited stores
    visited = set(interaction_matrix[cidx].nonzero()[1])
    for v in visited:
        scores[v] = -np.inf

    # Normalize scores to [0, 1]
    valid_mask = scores > -np.inf
    if not valid_mask.any():
        return []
    min_s, max_s = scores[valid_mask].min(), scores[valid_mask].max()
    if max_s > min_s:
        scores[valid_mask] = (scores[valid_mask] - min_s) / (max_s - min_s)
    else:
        scores[valid_mask] = 0.5

    # Sort by score descending
    ranked_indices = np.argsort(-scores)

    # Dedup by merchant+city
    merchant_rows = conn.execute(
        "SELECT store_id, merchant_name, city FROM store_profiles"
    ).fetchall()
    merchant_map = {r[0]: f"{r[1]}|{r[2]}" for r in merchant_rows if r[1]}

    # Build visited merchant keys to exclude
    visited_keys = set()
    for vi in visited:
        key = merchant_map.get(store_ids[vi], "")
        if key:
            visited_keys.add(key)

    seen_merchants = set(visited_keys)
    results = []
    for i in ranked_indices:
        if scores[i] <= -np.inf:
            break
        sid = store_ids[int(i)]
        merchant_key = merchant_map.get(sid, sid)
        if merchant_key in seen_merchants:
            continue
        seen_merchants.add(merchant_key)

        # Fetch store profile
        profile_row = conn.execute(
            "SELECT merchant_name, city, num_invoices, num_distinct_products, median_unit_price "
            "FROM store_profiles WHERE store_id = ?",
            (sid,),
        ).fetchone()
        profile = {}
        if profile_row:
            profile = {
                "merchant_name": profile_row[0],
                "city": profile_row[1],
                "num_invoices": profile_row[2],
                "num_products": profile_row[3],
                "median_price": profile_row[4],
            }

        results.append({
            "store_id": sid,
            "score": round(float(scores[int(i)]), 4),
            **profile,
        })
        if len(results) >= top_k:
            break

    return results


# ── Master Training Function ──────────────────────────────────────────


def train_all_models(conn: sqlite3.Connection, demo_dict: Optional[dict] = None) -> dict:
    """Train all model-based CF models and return them with shared data.

    Args:
        conn: SQLite connection with transaction and store profile data.
        demo_dict: Optional customer demographics dict from
            demographic.get_customer_demographics(conn).  When provided,
            also trains LightFM Demo (demographic user features only) and
            LightFM Full Hybrid (behavioral + demographic user features).
            If None, the demographic models are skipped gracefully.

    Returns dict with:
        interaction_matrix, customer_ids, store_ids,
        als_model, lightfm_model, lightfm_hybrid_model,
        item_features, user_features,
        (optional) lightfm_demo_model, demo_user_features,
        (optional) lightfm_full_hybrid_model, full_hybrid_user_features
    """
    print("Building interaction matrix for model-based CF...", flush=True)
    interaction_matrix, customer_ids, store_ids = _build_interaction_matrix(conn)
    print(f"  Matrix: {interaction_matrix.shape[0]} customers x {interaction_matrix.shape[1]} stores", flush=True)

    # ALS
    print("Training ALS model...", flush=True)
    als_model = train_als(interaction_matrix)

    # LightFM WARP (no features)
    print("Training LightFM WARP model...", flush=True)
    lightfm_model = train_lightfm(interaction_matrix)

    # LightFM Hybrid: item features + behavioral user features
    print("Building item features...", flush=True)
    item_features = _build_item_features(store_ids, conn)
    print("Building behavioral user features...", flush=True)
    user_features = _build_user_features(customer_ids, conn)
    print("Training LightFM Hybrid model (128 components, tuned)...", flush=True)
    lightfm_hybrid_model = train_lightfm(
        interaction_matrix,
        item_features=item_features,
        user_features=user_features,
        no_components=128,
        learning_rate=0.01,
        epochs=50,
    )

    result = {
        "interaction_matrix": interaction_matrix,
        "customer_ids": customer_ids,
        "store_ids": store_ids,
        "als_model": als_model,
        "lightfm_model": lightfm_model,
        "lightfm_hybrid_model": lightfm_hybrid_model,
        "item_features": item_features,
        "user_features": user_features,
    }

    # Demographic models — only if demo_dict was provided
    if demo_dict is not None:
        from pipelines.demographic import (
            _build_demographic_user_features,
            _build_full_hybrid_user_features,
        )

        # LightFM Demo: demographic user features + store item features.
        # Uses the same 64-component / lr=0.05 / 30-epoch setup as LightFM
        # WARP, keeping everything except the user representation identical
        # for a clean ablation: pure CF vs. demographics-augmented CF.
        print("Building demographic user features...", flush=True)
        demo_user_features = _build_demographic_user_features(customer_ids, demo_dict)
        print("Training LightFM Demo model (64 components, demographic features)...", flush=True)
        lightfm_demo_model = train_lightfm(
            interaction_matrix,
            item_features=item_features,
            user_features=demo_user_features,
            no_components=64,
            learning_rate=0.05,
            epochs=30,
        )

        # LightFM Full Hybrid: behavioral + demographic user features + store
        # item features.  Uses 128 components and slower learning rate to
        # accommodate the larger feature set (6 user features vs. 3).
        print("Building full hybrid user features (behavioral + demographic)...", flush=True)
        full_hybrid_user_features = _build_full_hybrid_user_features(
            customer_ids, conn, demo_dict
        )
        print("Training LightFM Full Hybrid model (128 components, all features)...", flush=True)
        lightfm_full_hybrid_model = train_lightfm(
            interaction_matrix,
            item_features=item_features,
            user_features=full_hybrid_user_features,
            no_components=128,
            learning_rate=0.01,
            epochs=50,
        )

        result.update({
            "lightfm_demo_model": lightfm_demo_model,
            "demo_user_features": demo_user_features,
            "lightfm_full_hybrid_model": lightfm_full_hybrid_model,
            "full_hybrid_user_features": full_hybrid_user_features,
        })
        print("Demographic models trained.", flush=True)

    print("All model-based CF models trained.", flush=True)
    return result
