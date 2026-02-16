#!/usr/bin/env python3
"""
Generate a quick QA/report for the scraped catalog SQLite DB.

Usage:
  python scripts/report.py --db data/catalog.sqlite --out reports/

Outputs:
  - merchant_summary.csv
  - image_stats.csv
  - samples_<merchant>.csv (N random)
  - scrape_report.html (simple visual card view)
"""
import argparse
import os
import sqlite3
import pandas as pd
import html

def html_escape(s):
    return html.escape("" if s is None else str(s))


def to_rel_web_path(path_from_db: str, report_dir: str) -> str:
    """Return an HTML-friendly relative path from the report folder to an image on disk.

    Handles common layouts:
      - stored as 'data/images/...'
      - stored as 'images/...' but actual files live in 'data/images/...'
      - stored as absolute path
    """
    if not path_from_db:
        return ""

    p = str(path_from_db).replace("\\", "/")

    # Resolve to an absolute path on disk
    if os.path.isabs(p):
        abs_img = p
    else:
        # First try as-is relative to project root (cwd)
        cand1 = os.path.abspath(p)
        if os.path.exists(cand1):
            abs_img = cand1
        else:
            # Common case: DB stored 'images/...', actual is 'data/images/...'
            cand2 = os.path.abspath(os.path.join("data", p))
            abs_img = cand2 if os.path.exists(cand2) else cand1

    # Make it relative to the report directory (so HTML works inside reports/)
    abs_report = os.path.abspath(report_dir)
    rel = os.path.relpath(abs_img, abs_report)
    return rel.replace(os.sep, "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/catalog.sqlite")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report_dir = os.path.abspath(args.out)  # <-- important: report folder path
    conn = sqlite3.connect(args.db)

    summary = pd.read_sql_query("""
    SELECT merchant,
           COUNT(*) AS products,
           SUM(CASE WHEN name IS NULL OR TRIM(name)='' THEN 1 ELSE 0 END) AS missing_name,
           SUM(CASE WHEN description IS NULL OR TRIM(description)='' THEN 1 ELSE 0 END) AS missing_description,
           SUM(CASE WHEN price IS NULL OR TRIM(price)='' THEN 1 ELSE 0 END) AS missing_price
    FROM products
    GROUP BY merchant
    ORDER BY products DESC;
    """, conn)
    summary.to_csv(os.path.join(args.out, "merchant_summary.csv"), index=False)

    img_stats = pd.read_sql_query("""
    SELECT p.merchant,
           COUNT(DISTINCT p.product_id) AS products,
           SUM(CASE WHEN i.cnt IS NULL OR i.cnt=0 THEN 1 ELSE 0 END) AS products_with_zero_images,
           AVG(COALESCE(i.cnt,0)) AS avg_images_per_product,
           MAX(COALESCE(i.cnt,0)) AS max_images_per_product
    FROM products p
    LEFT JOIN (
      SELECT product_id, COUNT(*) AS cnt
      FROM product_images
      GROUP BY product_id
    ) i ON i.product_id = p.product_id
    GROUP BY p.merchant
    ORDER BY products DESC;
    """, conn)
    img_stats.to_csv(os.path.join(args.out, "image_stats.csv"), index=False)

    sample_per = []
    for m in summary["merchant"].tolist():
        df = pd.read_sql_query(f"""
        SELECT p.product_id, p.name, p.price, p.currency, p.url,
               (SELECT local_path FROM product_images pi
                 WHERE pi.product_id=p.product_id
                 ORDER BY pi.image_url
                 LIMIT 1) AS first_image_path,
               (SELECT COUNT(*) FROM product_images pi
                 WHERE pi.product_id=p.product_id) AS num_images
        FROM products p
        WHERE p.merchant='{m}'
        ORDER BY RANDOM()
        LIMIT {int(args.samples)};
        """, conn)
        df.to_csv(os.path.join(args.out, f"samples_{m}.csv"), index=False)
        sample_per.append((m, df))

    # HTML report (IMPORTANT: image src paths must be relative to the reports/ folder)
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>Catalog Scrape Report</title>"
                 "<style>body{font-family:Arial, sans-serif; margin:24px;} table{border-collapse:collapse; width:100%; margin:12px 0;} "
                 "th,td{border:1px solid #ddd; padding:8px; font-size:14px;} th{background:#f5f5f5;} "
                 ".card{display:flex; gap:16px; border:1px solid #ddd; border-radius:8px; padding:12px; margin:10px 0;} "
                 ".img{width:140px; height:140px; object-fit:contain; background:#fafafa; border:1px solid #eee; border-radius:6px;} "
                 ".meta{flex:1;} .small{color:#666; font-size:12px;} </style></head><body>")
    parts.append("<h1>Catalog Scrape Report</h1>")
    parts.append(f"<p class='small'>DB: {html_escape(args.db)}</p>")
    parts.append("<h2>Merchant summary</h2>")
    parts.append(summary.to_html(index=False, escape=True))
    parts.append("<h2>Image stats</h2>")
    parts.append(img_stats.to_html(index=False, escape=True))

    for m, df in sample_per:
        parts.append(f"<h2>Samples: {html_escape(m)} ({len(df)} random products)</h2>")
        for _, r in df.iterrows():
            img_db = r.get("first_image_path") or ""
            img_rel = to_rel_web_path(img_db, report_dir) if img_db else ""   # <-- FIX

            name = r.get("name") or ""
            price = r.get("price") or ""
            currency = r.get("currency") or ""
            url = r.get("url") or ""
            num_images = int(r.get("num_images") or 0)

            parts.append("<div class='card'>")
            if img_rel:
                parts.append(f"<img class='img' src='{html_escape(img_rel)}' alt='image'/>")
            else:
                parts.append("<div class='img'></div>")
            parts.append("<div class='meta'>")
            parts.append(f"<div><b>{html_escape(name)}</b></div>")
            parts.append(f"<div>{html_escape(price)} {html_escape(currency)} &nbsp; <span class='small'>(images: {num_images})</span></div>")
            parts.append(f"<div class='small'>{html_escape(url)}</div>")
            parts.append("</div></div>")

    parts.append("</body></html>")
    with open(os.path.join(args.out, "scrape_report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print("Wrote report to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
