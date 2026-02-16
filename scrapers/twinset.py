"""Twinset scraper (category pagination -> product pages).

Usage:
  python -m scrapers.twinset --max-pages 5 --limit 500 --download-images
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from tqdm import tqdm

from .common import get, init_db, now_utc_iso, save_product_images, setup_logging, stable_id, upsert_product, already_scraped

logger = logging.getLogger(__name__)

TWINSET_BASE = "https://www.twinset.com"
TWINSET_SEED = "https://www.twinset.com/es-es/ropa/"
# Only match URLs ending with _<digits>.html (product code pattern)
TWINSET_PRODUCT_RE = re.compile(r"^https://www\.twinset\.com/es-es/.*_\d+\.html$")
PRICE_RE = re.compile(r"€\s*([0-9]+(?:[.,][0-9]{2})?)")


def twinset_extract_product_links(listing_html: str) -> list[str]:
    soup = BeautifulSoup(listing_html, "lxml")
    links = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        full = href if href.startswith("http") else urljoin(TWINSET_BASE, href)
        full = full.split("?")[0]
        if TWINSET_PRODUCT_RE.match(full):
            links.add(full)
    return sorted(links)


def twinset_product_urls(seed_url: str, max_pages: int = 3, sleep_s: float = 0.7) -> list[str]:
    all_links = set()
    for page in range(1, max_pages + 1):
        url = seed_url if page == 1 else f"{seed_url}?page={page}"
        html = get(url)
        links = twinset_extract_product_links(html)
        if not links:
            break
        all_links.update(links)
        logger.info("page %d: found %d product links (%d total)", page, len(links), len(all_links))
        time.sleep(sleep_s)
    return sorted(all_links)


def _extract_description(soup: BeautifulSoup) -> str | None:
    """Extract product description with multiple fallback strategies."""
    # Strategy 1: <meta name="description"> tag
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content", "").strip():
        content = meta["content"].strip()
        if len(content) > 20:
            return content

    # Strategy 2: known product-detail containers
    for selector in (
        '[itemprop="description"]',
        ".product-description",
        ".pdp-description",
        ".product-detail__description",
    ):
        el = soup.select_one(selector)
        if el:
            text = el.get_text(" ", strip=True)
            if len(text) > 20:
                return text

    # Strategy 3: give up rather than return wrong data
    return None


def parse_twinset_product(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    h1 = soup.find("h1")
    name = h1.get_text(" ", strip=True) if h1 else None

    m = PRICE_RE.search(text)
    price = m.group(1).replace(",", ".") if m else None
    currency = "EUR" if m else None

    desc = _extract_description(soup)

    image_urls = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "twinset-cdn.thron.com" in href:
            image_urls.append(href)
    for img in soup.select("img[src]"):
        src = img["src"]
        if "twinset-cdn.thron.com" in src:
            image_urls.append(src)

    seen = set()
    image_urls = [x for x in image_urls if not (x in seen or seen.add(x))]

    return {
        "url": url,
        "name": name,
        "description": desc,
        "price": price,
        "currency": currency,
        "image_urls": image_urls[:10],
    }


def main(max_pages: int, limit: int, download_images: bool, sleep_s: float = 0.5) -> None:
    conn = init_db()
    urls = twinset_product_urls(TWINSET_SEED, max_pages=max_pages)
    if limit:
        urls = urls[:limit]

    logger.info("scraping %d twinset product URLs", len(urls))

    for url in tqdm(urls, desc="twinset"):
        if already_scraped(conn, "twinset", url):
            continue
        html = get(url)
        p = parse_twinset_product(html, url)
        p["merchant"] = "twinset"
        p["product_id"] = stable_id(p["merchant"], url)
        p["scraped_at"] = now_utc_iso()

        upsert_product(conn, p)
        if download_images:
            save_product_images(conn, p["merchant"], p["product_id"], p.get("image_urls", []))

        time.sleep(sleep_s)

    logger.info("done twinset: %d products", len(urls))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--limit", type=int, default=500, help="Max number of product URLs to scrape (0 = no limit)")
    ap.add_argument("--download-images", action="store_true")
    args = ap.parse_args()

    setup_logging()
    main(max_pages=args.max_pages, limit=args.limit, download_images=args.download_images)
