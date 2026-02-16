"""Small helper to inspect what you've scraped (SQLite -> console tables)."""

import logging
import sqlite3
import sys

import pandas as pd

# Allow running as standalone script
sys.path.insert(0, ".")
from scrapers.common import setup_logging

logger = logging.getLogger(__name__)

DB = "data/catalog.sqlite"

def main():
    conn = sqlite3.connect(DB)
    summary = pd.read_sql_query(
        """SELECT merchant, COUNT(*) AS products
           FROM products
           GROUP BY merchant
           ORDER BY products DESC""",
        conn,
    )
    logger.info("Product counts by merchant:\n%s", summary.to_string(index=False))

    quality = pd.read_sql_query(
        """SELECT merchant,
              SUM(CASE WHEN name IS NULL OR TRIM(name)='' THEN 1 ELSE 0 END) AS missing_name,
              SUM(CASE WHEN price IS NULL OR TRIM(price)='' THEN 1 ELSE 0 END) AS missing_price,
              COUNT(*) AS total
            FROM products
            GROUP BY merchant""",
        conn,
    )
    logger.info("Field completeness:\n%s", quality.to_string(index=False))

    top = pd.read_sql_query(
        """SELECT p.merchant, p.product_id, p.name, p.price, p.currency, p.url,
              COUNT(pi.image_url) AS num_images
            FROM products p
            LEFT JOIN product_images pi ON pi.product_id = p.product_id
            GROUP BY p.merchant, p.product_id
            ORDER BY num_images DESC
            LIMIT 20""",
        conn,
    )
    logger.info("Top products by downloaded images:\n%s", top.to_string(index=False))

if __name__ == "__main__":
    setup_logging()
    main()
