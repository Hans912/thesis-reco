"""Arcaplanet scraper (sitemap-driven).

Usage:
  python -m scrapers.arcaplanet --max-sitemaps 2 --limit 500 --download-images
"""

from __future__ import annotations

import logging
import time
from typing import List

import extruct
from bs4 import BeautifulSoup
from w3lib.html import get_base_url
from tqdm import tqdm

from .common import get, init_db, now_utc_iso, save_product_images, setup_logging, stable_id, upsert_product, already_scraped

logger = logging.getLogger(__name__)


def parse_sitemap_urls(xml_text: str) -> list[str]:
    soup = BeautifulSoup(xml_text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]


def arcaplanet_product_urls(max_sitemaps: int = 3, sleep_s: float = 0.5) -> list[str]:
    urls: list[str] = []
    for i in range(max_sitemaps):
        sm_url = f"https://www.arcaplanet.it/sitemap/product-{i}.xml"
        xml = get(sm_url)
        urls.extend(parse_sitemap_urls(xml))
        time.sleep(sleep_s)
    urls = [u.split("?")[0] for u in urls if u.endswith("/p")]
    return sorted(set(urls))


def extract_jsonld_product(html: str, url: str) -> dict | None:
    data = extruct.extract(html, base_url=get_base_url(html, url), syntaxes=["json-ld"])
    for obj in data.get("json-ld", []):
        if isinstance(obj, dict) and obj.get("@type") in ("Product", ["Product"]):
            return obj
    return None


def parse_arcaplanet_product(html: str, url: str) -> dict:
    prod = extract_jsonld_product(html, url)
    soup = BeautifulSoup(html, "lxml")

    name = None
    desc = None
    price = None
    currency = None
    images: list[str] = []

    if prod:
        name = prod.get("name")
        desc = prod.get("description")
        offers = prod.get("offers") or {}
        if isinstance(offers, list) and offers:
            offers = offers[0]
        price = offers.get("price")
        currency = offers.get("priceCurrency")
        img = prod.get("image")
        if isinstance(img, list):
            images = img
        elif isinstance(img, str):
            images = [img]

    if not name:
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(" ", strip=True)

    return {
        "url": url,
        "name": name,
        "description": desc,
        "price": str(price) if price is not None else None,
        "currency": currency,
        "image_urls": images[:10],
    }


def main(max_sitemaps: int, limit: int, download_images: bool, sleep_s: float = 0.5) -> None:
    conn = init_db()
    urls = arcaplanet_product_urls(max_sitemaps=max_sitemaps)
    if limit:
        urls = urls[:limit]

    logger.info("scraping %d arcaplanet product URLs", len(urls))

    for url in tqdm(urls, desc="arcaplanet"):
        if already_scraped(conn, "arcaplanet", url):
            continue
        html = get(url)
        p = parse_arcaplanet_product(html, url)
        p["merchant"] = "arcaplanet"
        p["product_id"] = stable_id(p["merchant"], url)
        p["scraped_at"] = now_utc_iso()

        upsert_product(conn, p)
        if download_images:
            save_product_images(conn, p["merchant"], p["product_id"], p.get("image_urls", []))

        time.sleep(sleep_s)

    logger.info("done arcaplanet: %d products", len(urls))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sitemaps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=500, help="Max number of product URLs to scrape (0 = no limit)")
    ap.add_argument("--download-images", action="store_true")
    args = ap.parse_args()

    setup_logging()
    main(max_sitemaps=args.max_sitemaps, limit=args.limit, download_images=args.download_images)
