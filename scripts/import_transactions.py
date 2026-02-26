"""Import transaction and store profile CSVs into the catalog SQLite database."""

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "catalog.sqlite"
TRANSACTIONS_CSV = ROOT / "data" / "final_transaction_data.csv"
STORE_PROFILES_CSV = ROOT / "data" / "final_store_profile_transaction.csv"


def init_transaction_tables(conn: sqlite3.Connection) -> None:
    """Create transaction-related tables if they don't exist."""
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS transactions(
        invoice_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        line_num INTEGER NOT NULL DEFAULT 0,
        customer_id TEXT NOT NULL,
        store_id TEXT NOT NULL,
        merchant_id TEXT NOT NULL,
        description TEXT,
        qty REAL,
        unit_price REAL,
        vat_rate REAL,
        issued_on TEXT,
        PRIMARY KEY(invoice_id, product_id, line_num)
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS store_profiles(
        store_id TEXT PRIMARY KEY,
        merchant_id TEXT NOT NULL,
        merchant_name TEXT,
        city TEXT,
        revenue REAL,
        num_invoices INTEGER,
        num_distinct_products INTEGER,
        median_unit_price REAL
    )""")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_store ON transactions(store_id)")

    conn.commit()


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
                   (invoice_id, product_id, line_num, customer_id, store_id, merchant_id,
                    description, qty, unit_price, vat_rate, issued_on)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["invoice_id"],
                    row["product_id"],
                    int(row["line_num"]) if row.get("line_num") else 0,
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

    # Ensure scraping tables exist (products, product_images, stores, favorites)
    from scrapers.common import init_db
    init_db(str(DB_PATH))

    # Ensure transaction tables exist
    init_transaction_tables(conn)

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
