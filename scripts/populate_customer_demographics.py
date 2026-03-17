"""Populate customer_demographics table in SQLite from external data sources.

Joins:
  - data/customer_info.xlsx  → DateOfBirth per customer
  - data/final_transaction_data.csv → nationality (passport country) and
    residency country per customer

Computes per customer:
  - age (integer, clipped to [16, 100])
  - age_bin (18-25 | 26-35 | 36-50 | 51-65 | 65+)
  - nationality (ISO2 passport country code)
  - residency (ISO2 residency country code)
  - tourist_type (cross_border | domestic | international)

Run once before training or evaluation:
    python -m scripts.populate_customer_demographics
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.demographic import _compute_age_bin, _compute_tourist_type

DB_PATH = ROOT / "data" / "catalog.sqlite"
CUSTOMER_INFO_PATH = ROOT / "data" / "customer_info.xlsx"
TRANSACTIONS_CSV_PATH = ROOT / "data" / "final_transaction_data.csv"

REFERENCE_DATE = pd.Timestamp("2026-03-12")  # current date for age computation


def main():
    print("Loading customer_info.xlsx…", flush=True)
    info_df = pd.read_excel(CUSTOMER_INFO_PATH)
    info_df.columns = [c.split(".")[-1] for c in info_df.columns]  # strip prefix
    info_df = info_df.rename(columns={
        "CustomerId": "customer_id",
        "DateOfBirth": "dob",
    })
    info_df["customer_id"] = info_df["customer_id"].str.lower().str.strip()
    info_df["dob"] = pd.to_datetime(info_df["dob"], errors="coerce")
    print(f"  {len(info_df)} customers in info file", flush=True)

    print("Loading demographic columns from transaction CSV…", flush=True)
    # Only load the columns we need — the CSV is 754k rows but we only need
    # one row per customer for nationality and residency.
    txn_df = pd.read_csv(
        TRANSACTIONS_CSV_PATH,
        usecols=["CustomerId", "CustomerPassportIssuingCountryIso2Code",
                 "CustomerResidencyCountryIso2Code"],
        low_memory=False,
    )
    txn_df["customer_id"] = txn_df["CustomerId"].str.lower().str.strip()
    txn_demo = (
        txn_df
        .groupby("customer_id")
        .first()
        .reset_index()
        [["customer_id", "CustomerPassportIssuingCountryIso2Code",
          "CustomerResidencyCountryIso2Code"]]
        .rename(columns={
            "CustomerPassportIssuingCountryIso2Code": "nationality",
            "CustomerResidencyCountryIso2Code": "residency",
        })
    )
    print(f"  {len(txn_demo)} unique customers in transaction CSV", flush=True)

    # Join on customer_id
    merged = info_df.merge(txn_demo, on="customer_id", how="inner")
    print(f"  {len(merged)} customers after join", flush=True)

    # Compute age — clip outliers to realistic range [16, 100]
    merged["age"] = (REFERENCE_DATE - merged["dob"]).dt.days // 365
    merged["age"] = merged["age"].clip(16, 100)
    merged["age"] = merged["age"].where(merged["dob"].notna(), other=None)

    # Compute derived demographic features
    merged["age_bin"] = merged["age"].apply(
        lambda a: _compute_age_bin(int(a)) if pd.notna(a) else "age:unknown"
    )
    merged["tourist_type"] = merged["residency"].apply(
        lambda r: _compute_tourist_type(str(r) if pd.notna(r) else "")
    )
    merged["nationality"] = merged["nationality"].fillna("").str.upper().str.strip()
    merged["residency"] = merged["residency"].fillna("").str.upper().str.strip()

    # Write to SQLite
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_demographics (
            customer_id  TEXT PRIMARY KEY,
            age          INTEGER,
            age_bin      TEXT,
            nationality  TEXT,
            residency    TEXT,
            tourist_type TEXT
        )
    """)
    conn.execute("DELETE FROM customer_demographics")  # full refresh

    rows = [
        (
            row["customer_id"],
            int(row["age"]) if pd.notna(row["age"]) else None,
            row["age_bin"],
            row["nationality"] or None,
            row["residency"] or None,
            row["tourist_type"],
        )
        for _, row in merged.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO customer_demographics "
        "(customer_id, age, age_bin, nationality, residency, tourist_type) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM customer_demographics").fetchone()[0]
    print(f"\nInserted {total} customer demographic records.", flush=True)

    print("\nAge bin distribution:")
    for row in conn.execute(
        "SELECT age_bin, COUNT(*) FROM customer_demographics GROUP BY age_bin ORDER BY age_bin"
    ).fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\nTourist type distribution:")
    for row in conn.execute(
        "SELECT tourist_type, COUNT(*) FROM customer_demographics GROUP BY tourist_type"
    ).fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\nTop 10 nationalities:")
    for row in conn.execute(
        "SELECT nationality, COUNT(*) as n FROM customer_demographics "
        "GROUP BY nationality ORDER BY n DESC LIMIT 10"
    ).fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
