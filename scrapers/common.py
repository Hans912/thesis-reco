"""Common utilities for merchant scrapers (HTTP, parsing helpers, SQLite storage, image downloads).

Designed for a thesis prototype:
- polite requests (rate limit outside)
- robust retries
- SQLite as local catalog store
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    # Override in your scripts if you want a contact email or a different UA.
    "User-Agent": "Mozilla/5.0 (thesis scraper; contact: you@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_shared_session: Optional[requests.Session] = None


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stable_id(merchant: str, url: str) -> str:
    """Stable short id for a product based on merchant + URL."""
    return hashlib.sha256(f"{merchant}::{url}".encode("utf-8")).hexdigest()[:24]


def init_db(db_path="data/catalog.sqlite"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS products(
        product_id TEXT PRIMARY KEY,
        merchant TEXT,
        url TEXT,
        name TEXT,
        description TEXT,
        price TEXT,
        currency TEXT,
        scraped_at TEXT
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS product_images(
        product_id TEXT,
        image_url TEXT,
        local_path TEXT,
        PRIMARY KEY(product_id, image_url)
    )""")

    # NEW: speed + uniqueness protection
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_products_merchant_url ON products(merchant, url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_products_merchant_scraped_at ON products(merchant, scraped_at)")

    conn.commit()
    return conn



def upsert_product(conn: sqlite3.Connection, p: dict) -> None:
    cur = conn.cursor()
    cur.execute(
        """INSERT OR REPLACE INTO products
           (product_id, merchant, url, name, description, price, currency, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            p["product_id"],
            p["merchant"],
            p["url"],
            p.get("name"),
            p.get("description"),
            p.get("price"),
            p.get("currency"),
            p["scraped_at"],
        ),
    )
    conn.commit()


def get_session(headers: Optional[dict] = None) -> requests.Session:
    """Return a module-level cached session (creates once, reuses thereafter)."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        _shared_session.headers.update(DEFAULT_HEADERS)
    if headers:
        _shared_session.headers.update(headers)
    return _shared_session


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get(url: str, *, headers: Optional[dict] = None, timeout_s: int = 30) -> str:
    """HTTP GET with retries. Returns response text."""
    s = get_session(headers)
    r = s.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def download_image(image_url: str, out_path: str, *, headers: Optional[dict] = None, timeout_s: int = 30) -> None:
    """Download an image and save as JPEG."""
    s = get_session(headers)
    r = s.get(image_url, timeout=timeout_s)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)


def save_product_images(
    conn: sqlite3.Connection,
    merchant: str,
    product_id: str,
    image_urls: list[str],
    *,
    images_root: str = "data/images",
    sleep_s: float = 0.2,
    headers: Optional[dict] = None,
) -> None:
    """Download multiple images for a product and record them in SQLite."""
    if not image_urls:
        return

    # de-dupe preserving order
    seen = set()
    image_urls = [u for u in image_urls if u and not (u in seen or seen.add(u))]

    cur = conn.cursor()
    out_dir = Path(images_root) / merchant / product_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_url in enumerate(image_urls):
        local_path = out_dir / f"{idx}.jpg"

        # skip if already recorded and file exists
        cur.execute(
            "SELECT 1 FROM product_images WHERE product_id=? AND image_url=?",
            (product_id, img_url),
        )
        if cur.fetchone() and local_path.exists():
            continue

        try:
            download_image(img_url, str(local_path), headers=headers)
            cur.execute(
                "INSERT OR REPLACE INTO product_images(product_id, image_url, local_path) VALUES (?, ?, ?)",
                (product_id, img_url, str(local_path)),
            )
            conn.commit()
        except Exception as e:
            logger.warning("image fail  merchant=%s  product=%s  idx=%d  url=%s  err=%s", merchant, product_id, idx, img_url, e)

        time.sleep(sleep_s)


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat()


def already_scraped(conn, merchant: str, url: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM products WHERE merchant=? AND url=? LIMIT 1", (merchant, url))
    return cur.fetchone() is not None
