"""Import transaction and store profile CSVs into the catalog SQLite database."""

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "catalog.sqlite"
TRANSACTIONS_CSV = ROOT / "data" / "test_month_transaction_product.csv"
STORE_PROFILES_CSV = ROOT / "data" / "test_store_location_transaction.csv"


def import_transactions(conn: sqlite3.Connection) -> int:
    """Import transaction items from CSV into the transactions table."""
    if not TRANSACTIONS_CSV.exists():
        print(f"Transaction CSV not found: {TRANSACTIONS_CSV}")
        return 0

    cur = conn.cursor()
    count = 0
    with open(TRANSACTIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(
                """INSERT OR IGNORE INTO transactions
                   (invoice_id, product_id, customer_id, store_id, merchant_id,
                    description, qty, unit_price, vat_rate, issued_on)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["invoice_id"],
                    row["product_id"],
                    row["CustomerId"],
                    row["StoreId"],
                    row["MerchantId"],
                    row["desc_clean"],
                    float(row["qty"]) if row["qty"] else None,
                    float(row["unit_price_with_vat"]) if row["unit_price_with_vat"] else None,
                    float(row["vat_rate"]) if row["vat_rate"] else None,
                    row["IssuedOn"],
                ),
            )
            count += 1
    conn.commit()
    return count


def import_store_profiles(conn: sqlite3.Connection) -> int:
    """Import store profile summaries from CSV into store_profiles table."""
    if not STORE_PROFILES_CSV.exists():
        print(f"Store profiles CSV not found: {STORE_PROFILES_CSV}")
        return 0

    cur = conn.cursor()
    count = 0
    with open(STORE_PROFILES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(
                """INSERT OR REPLACE INTO store_profiles
                   (store_id, merchant_id, merchant_name, city, revenue,
                    num_invoices, num_distinct_products, median_unit_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["StoreId"],
                    row["MerchantId"],
                    row.get("merchant_name"),
                    row.get("MerchantCity"),
                    float(row["revenue_items_sum"]) if row.get("revenue_items_sum") else None,
                    int(row["num_invoices"]) if row.get("num_invoices") else None,
                    int(row["num_distinct_products"]) if row.get("num_distinct_products") else None,
                    float(row["median_unit_price"]) if row.get("median_unit_price") else None,
                ),
            )
            count += 1
    conn.commit()
    return count


if __name__ == "__main__":
    conn = sqlite3.connect(str(DB_PATH))

    # Ensure tables exist
    from scrapers.common import init_db
    init_db(str(DB_PATH))

    print("Importing transactions...")
    n_txn = import_transactions(conn)
    print(f"  Imported {n_txn} transaction items.")

    print("Importing store profiles...")
    n_sp = import_store_profiles(conn)
    print(f"  Imported {n_sp} store profiles.")

    # Quick summary
    cur = conn.cursor()
    n_customers = cur.execute("SELECT COUNT(DISTINCT customer_id) FROM transactions").fetchone()[0]
    n_invoices = cur.execute("SELECT COUNT(DISTINCT invoice_id) FROM transactions").fetchone()[0]
    n_stores = cur.execute("SELECT COUNT(*) FROM store_profiles").fetchone()[0]
    print(f"\nSummary: {n_txn} items, {n_invoices} invoices, {n_customers} customers, {n_stores} store profiles")

    conn.close()
