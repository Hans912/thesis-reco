"""Demographic-based recommendation models.

Two models are implemented:

1. Demographic Popularity (segment-based baseline)
   For each customer, identifies their demographic segment
   (nationality × age_bin × tourist_type) and recommends the stores most
   visited by other customers in that segment during the training period.
   Falls back gracefully through coarser segments when the target segment
   has too few customers to produce a reliable popularity signal.
   This is a pure cold-start model — no behavioral history is required,
   making it applicable to brand-new users arriving with only passport data.

2. LightFM Demo (demographic user features, no behavioral features)
   Trains LightFM WARP on the standard interaction matrix but represents
   users through demographic features only (nationality, age_bin,
   tourist_type) rather than behavioral features (visit frequency, spend,
   primary city).  Store item features (region, size, price) are included.
   This isolates the predictive power of demographic signals alone, and
   serves as the direct comparison against LightFM Hybrid which uses
   behavioral features with the same store item features.

3. LightFM Full Hybrid (behavioral + demographic user features)
   Extends the existing LightFM Hybrid by adding nationality, age_bin, and
   tourist_type alongside the behavioral features (visit frequency, spend
   level, primary shopping city).  This tests whether demographic features
   improve upon behavioral CF and is the target production model.

Tourist type segmentation (cross_border / domestic / international) is
specific to this tax-free tourist shopping dataset: ~72 % of customers are
Swiss residents doing repeated cross-border shopping in Italy, which is a
distinct behavioural pattern from one-off international tourists.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────

# Countries whose residents do regular cross-border shopping in Italy —
# typically neighbouring countries or those with very close geographic/
# cultural ties.  Swiss residents (CH) account for ~72 % of this dataset.
CROSS_BORDER_COUNTRIES = {"CH", "AT", "FR", "SI", "SM", "HR", "ME", "AL"}

# Minimum number of distinct customers a segment must contain before its
# store popularity counts are considered reliable.  Below this threshold the
# fallback hierarchy is applied (nationality+age → nationality → global).
MIN_SEGMENT_SIZE = 20


# ── Demographic feature helpers ───────────────────────────────────────────


def _compute_tourist_type(residency: str) -> str:
    """Map residency ISO2 country code to tourist_type label.

    domestic      — Italian residents shopping at home (IT)
    cross_border  — residents of neighbouring countries doing regular
                    cross-border shopping (CH, AT, FR, SI, SM, ...)
    international — long-haul tourists (US, CN, UK, SA, ...)
    """
    if not residency:
        return "tourist:international"
    r = residency.strip().upper()
    if r == "IT":
        return "tourist:domestic"
    if r in CROSS_BORDER_COUNTRIES:
        return "tourist:cross_border"
    return "tourist:international"


def _compute_age_bin(age: int) -> str:
    """Bin integer age into standard demographic cohorts."""
    if age <= 25:
        return "age:18-25"
    elif age <= 35:
        return "age:26-35"
    elif age <= 50:
        return "age:36-50"
    elif age <= 65:
        return "age:51-65"
    else:
        return "age:65+"


# ── Database access ───────────────────────────────────────────────────────


def get_customer_demographics(conn: sqlite3.Connection) -> dict:
    """Load customer demographics from the customer_demographics table.

    Returns a dict: customer_id → {nationality, age_bin, tourist_type}
    where feature values are pre-formatted as LightFM feature strings
    (e.g. "nat:CH", "age:36-50", "tourist:cross_border").

    Requires the table to have been populated by:
        python -m scripts.populate_customer_demographics
    """
    rows = conn.execute(
        "SELECT customer_id, nationality, age_bin, tourist_type "
        "FROM customer_demographics"
    ).fetchall()
    return {
        r[0]: {
            "nationality": f"nat:{r[1]}" if r[1] else "nat:unknown",
            "age_bin": r[2] if r[2] else "age:unknown",
            "tourist_type": r[3] if r[3] else "tourist:international",
        }
        for r in rows
    }


# ── LightFM feature builders ──────────────────────────────────────────────


def _build_demographic_user_features(
    customer_ids: list[str],
    demo_dict: dict,
):
    """Build LightFM user feature matrix from demographic data only.

    Each user is represented by three categorical features:
      - nationality  (ISO2 passport country, e.g. "nat:CH")
      - age_bin      (cohort, e.g. "age:36-50")
      - tourist_type ("tourist:cross_border" | "tourist:domestic" |
                      "tourist:international")

    No behavioral data is used.  Combined with store item features, this
    allows the model to learn associations between demographic profiles and
    store characteristics (region, size, price tier) without relying on
    any purchase history — enabling genuine cold-start inference.

    The integer indices in the LightFM Dataset map 1-to-1 with the
    customer_ids list, matching the convention used throughout collab_model.py.
    """
    from lightfm.data import Dataset

    all_features: set[str] = set()
    for cid in customer_ids:
        d = demo_dict.get(cid, {})
        all_features.add(d.get("nationality", "nat:unknown"))
        all_features.add(d.get("age_bin", "age:unknown"))
        all_features.add(d.get("tourist_type", "tourist:international"))

    dataset = Dataset()
    dataset.fit(
        users=range(len(customer_ids)),
        items=range(1),  # dummy — we only need user features here
        user_features=sorted(all_features),
    )

    user_feature_list = []
    for i, cid in enumerate(customer_ids):
        d = demo_dict.get(cid, {})
        feats = [
            d.get("nationality", "nat:unknown"),
            d.get("age_bin", "age:unknown"),
            d.get("tourist_type", "tourist:international"),
        ]
        user_feature_list.append((i, feats))

    return dataset.build_user_features(user_feature_list)


def _build_full_hybrid_user_features(
    customer_ids: list[str],
    conn: sqlite3.Connection,
    demo_dict: dict,
    before_date: Optional[str] = None,
):
    """Build LightFM user features combining behavioral and demographic signals.

    Each user is represented by six features:
      Behavioral (3):
        freq:single/casual/regular/power  — visit frequency bin
        spend:low/mid/high                — total spend bin (P33/P66 thresholds)
        ucity:<city>                      — primary shopping city

      Demographic (3):
        nat:<ISO2>                        — nationality
        age:<bin>                         — age cohort
        tourist:<type>                    — tourist type

    Spend thresholds match collab_model._build_user_features (290/890 EUR,
    empirical P33/P66 of the full customer distribution).

    This feature set gives the model both what the user has done (behavioral)
    and who they are (demographic), enabling it to learn interactions between
    the two — e.g., "high-spending Swiss cross-border shoppers aged 36-50
    prefer mid-to-large stores in the Lake Como region."
    """
    from lightfm.data import Dataset

    date_filter = "WHERE issued_on < ?" if before_date else ""
    params = (before_date,) if before_date else ()

    profiles = {}
    for cid in customer_ids:
        # ── Behavioral features ──────────────────────────────────────────
        row = conn.execute(
            f"SELECT COUNT(DISTINCT store_id), SUM(qty * unit_price) "
            f"FROM transactions {date_filter} "
            f"{'AND' if before_date else 'WHERE'} customer_id = ?",
            params + (cid,),
        ).fetchone()
        n_stores = row[0] or 0
        total_spend = row[1] or 0

        city_row = conn.execute(
            f"SELECT sp.city, COUNT(*) as cnt "
            f"FROM transactions t JOIN store_profiles sp ON t.store_id = sp.store_id "
            f"{'WHERE t.issued_on < ?' if before_date else ''} "
            f"{'AND' if before_date else 'WHERE'} t.customer_id = ? "
            f"GROUP BY sp.city ORDER BY cnt DESC LIMIT 1",
            params + (cid,),
        ).fetchone()
        primary_city = (
            city_row[0].strip().title() if city_row and city_row[0] else "unknown"
        )

        if n_stores <= 1:
            freq = "freq:single"
        elif n_stores <= 3:
            freq = "freq:casual"
        elif n_stores <= 8:
            freq = "freq:regular"
        else:
            freq = "freq:power"

        if total_spend < 290:
            spend = "spend:low"
        elif total_spend < 890:
            spend = "spend:mid"
        else:
            spend = "spend:high"

        # ── Demographic features ─────────────────────────────────────────
        d = demo_dict.get(cid, {})

        profiles[cid] = {
            "freq": freq,
            "spend": spend,
            "city": f"ucity:{primary_city}",
            "nationality": d.get("nationality", "nat:unknown"),
            "age_bin": d.get("age_bin", "age:unknown"),
            "tourist_type": d.get("tourist_type", "tourist:international"),
        }

    all_features: set[str] = set()
    for p in profiles.values():
        all_features.update(p.values())

    dataset = Dataset()
    dataset.fit(
        users=range(len(customer_ids)),
        items=range(1),  # dummy
        user_features=sorted(all_features),
    )

    user_feature_list = [
        (i, list(profiles[cid].values()))
        for i, cid in enumerate(customer_ids)
    ]

    return dataset.build_user_features(user_feature_list)


# ── Live API recommendation ───────────────────────────────────────────────


def recommend_demographic_popularity(
    customer_id: str,
    conn: sqlite3.Connection,
    demo_dict: dict,
    store_ids: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Recommend stores using the Demographic Popularity model (live API).

    Identifies the customer's demographic segment and returns the most-visited
    stores within that segment from the full training history.  Applies the
    fallback hierarchy when a segment is too sparse.

    This function is designed for the live API where we have the full training
    data available (no temporal split).  Evaluation uses a separate function
    in evaluation.py that respects the train/test split.
    """
    store_set = set(store_ids)

    # Customer's visited stores (to exclude from recommendations)
    visited = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT store_id FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchall()
    )

    # Customer's demographics
    d = demo_dict.get(customer_id, {})
    nat = d.get("nationality", "nat:unknown")
    age = d.get("age_bin", "age:unknown")
    tourist = d.get("tourist_type", "tourist:international")
    full_seg = (nat, age, tourist)
    nat_age_seg = (nat, age)

    # Build segment store counts from ALL training data
    rows = conn.execute(
        "SELECT t.customer_id, t.store_id, COUNT(DISTINCT t.invoice_id) "
        "FROM transactions t GROUP BY t.customer_id, t.store_id"
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
        cd = demo_dict.get(cid, {})
        cn = cd.get("nationality", "nat:unknown")
        ca = cd.get("age_bin", "age:unknown")
        ct = cd.get("tourist_type", "tourist:international")

        seg_full[(cn, ca, ct)][sid] += visits
        seg_full_n[(cn, ca, ct)].add(cid)
        seg_nat_age[(cn, ca)][sid] += visits
        seg_nat_age_n[(cn, ca)].add(cid)
        seg_nat[cn][sid] += visits
        seg_nat_n[cn].add(cid)
        global_counts[sid] += visits

    def rank(counts):
        return [
            s for s, _ in sorted(counts.items(), key=lambda x: -x[1])
            if s in store_set and s not in visited
        ]

    # Select segment by fallback hierarchy
    if len(seg_full_n[full_seg]) >= MIN_SEGMENT_SIZE:
        ranked = rank(seg_full[full_seg])
        segment_used = f"{nat}+{age}+{tourist}"
    elif len(seg_nat_age_n[nat_age_seg]) >= MIN_SEGMENT_SIZE:
        ranked = rank(seg_nat_age[nat_age_seg])
        segment_used = f"{nat}+{age}"
    elif len(seg_nat_n[nat]) >= MIN_SEGMENT_SIZE:
        ranked = rank(seg_nat[nat])
        segment_used = nat
    else:
        ranked = rank(global_counts)
        segment_used = "global"

    results = []
    max_score = len(ranked) or 1
    for i, sid in enumerate(ranked[:top_k]):
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
        # Score: inversely proportional to rank (1.0 for rank 1, decaying)
        score = round(1.0 - (i / max_score), 4)
        results.append({"store_id": sid, "score": score, "segment": segment_used, **profile})

    return results
