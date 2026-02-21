"""Import store locations from an Excel or CSV file into SQLite.

Usage:
    python scripts/import_stores.py data/stores.xlsx
    python scripts/import_stores.py data/stores.csv

Expected columns (flexible naming — see COLUMN_MAP):
    StoreId, CompanyName, DisplayName,
    Coordinates (JSON with lat/lng), StreetName, StreetNumber, ZipCode,
    GooglePlaceId
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "catalog.sqlite"

# Map CompanyName substrings → our merchant slug.
# Add entries here as the catalog expands.
COMPANY_TO_MERCHANT = {
    "agrifarma": "arcaplanet",
    "arcaplanet": "arcaplanet",
    "iper animal": "arcaplanet",
    "iperanimal": "arcaplanet",
    "twinset": "twinset",
}


def resolve_merchant(company_name: str | None, display_name: str | None) -> str | None:
    """Map a company/display name to our merchant slug."""
    for text in [company_name, display_name]:
        if not text:
            continue
        lower = text.lower()
        for pattern, merchant in COMPANY_TO_MERCHANT.items():
            if pattern in lower:
                return merchant
    return None


def parse_coordinates(raw: str | dict | None) -> tuple[float, float] | None:
    """Extract (lat, lng) from the JSON coordinates column."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None

    if isinstance(raw, dict):
        points = raw.get("points", [])
        if points:
            p = points[0]
            lat = p.get("lat") or p.get("x")
            lng = p.get("lng") or p.get("y")
            if lat is not None and lng is not None:
                return float(lat), float(lng)

    return None


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive, partial match)."""
    cols_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols_lower:
            return cols_lower[key]
        for col_key, col_orig in cols_lower.items():
            if key in col_key:
                return col_orig
    return None


def import_stores(file_path: str, db_path: str = str(DB_PATH)) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Read file
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        print(f"Unsupported file type: {path.suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(df)} rows from {path.name}")
    print(f"Columns: {list(df.columns)}")

    # Resolve column names flexibly
    col_store_id = find_column(df, ["StoreId", "store_id"])
    col_company = find_column(df, ["CompanyName", "company_name"])
    col_display = find_column(df, ["DisplayName", "display_name"])
    col_coords = find_column(df, ["Coordinates", "coordinates"])
    col_street = find_column(df, ["StreetName", "street_name", "street"])
    col_street_num = find_column(df, ["StreetNumber", "street_number"])
    col_zip = find_column(df, ["ZipCode", "zip_code", "postal"])
    col_place_id = find_column(df, ["GooglePlaceId", "google_place_id", "place_id"])

    if not col_coords:
        print("Could not find a Coordinates column.", file=sys.stderr)
        sys.exit(1)

    # Open DB and ensure stores table exists
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS stores(
        store_id TEXT PRIMARY KEY,
        merchant TEXT NOT NULL,
        display_name TEXT,
        company_name TEXT,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        street TEXT,
        street_number TEXT,
        zip_code TEXT,
        google_place_id TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stores_merchant ON stores(merchant)")

    imported = 0
    skipped_no_merchant = 0
    skipped_no_coords = 0

    for _, row in df.iterrows():
        company = str(row.get(col_company, "")) if col_company else None
        display = str(row.get(col_display, "")) if col_display else None

        merchant = resolve_merchant(company, display)
        if not merchant:
            skipped_no_merchant += 1
            continue

        coords = parse_coordinates(row.get(col_coords) if col_coords else None)
        if not coords:
            skipped_no_coords += 1
            continue

        lat, lng = coords
        store_id = str(row.get(col_store_id, "")) if col_store_id else f"{merchant}_{lat}_{lng}"

        conn.execute(
            """INSERT OR REPLACE INTO stores
               (store_id, merchant, display_name, company_name, lat, lng,
                street, street_number, zip_code, google_place_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                store_id,
                merchant,
                display if display and display != "nan" else None,
                company if company and company != "nan" else None,
                lat,
                lng,
                str(row.get(col_street, "")) if col_street else None,
                str(row.get(col_street_num, "")) if col_street_num else None,
                str(row.get(col_zip, "")) if col_zip else None,
                str(row.get(col_place_id, "")) if col_place_id else None,
            ),
        )
        imported += 1

    conn.commit()

    # Summary
    counts = conn.execute(
        "SELECT merchant, COUNT(*) FROM stores GROUP BY merchant"
    ).fetchall()
    conn.close()

    print(f"\nImported {imported} stores")
    if skipped_no_merchant:
        print(f"Skipped {skipped_no_merchant} rows (unrecognized merchant)")
    if skipped_no_coords:
        print(f"Skipped {skipped_no_coords} rows (missing coordinates)")
    print("\nStores by merchant:")
    for merchant, count in counts:
        print(f"  {merchant}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Import store locations into SQLite")
    parser.add_argument("file", help="Path to Excel (.xlsx) or CSV (.csv) file")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path")
    args = parser.parse_args()
    import_stores(args.file, args.db)


if __name__ == "__main__":
    main()
