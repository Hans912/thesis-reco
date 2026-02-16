# Thesis scrapers (Arcaplanet + Twinset)

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run scrapers
```bash
python -m scrapers.arcaplanet --max-sitemaps 2 --limit 500 --download-images
python -m scrapers.twinset --max-pages 5 --limit 500 --download-images
```

Artifacts:
- `data/catalog.sqlite`
- `data/images/<merchant>/<product_id>/<idx>.jpg`

## Inspect scraped data
```bash
python scripts/summary.py
```

## View SQLite in a GUI
Install **DB Browser for SQLite** (free) and open `data/catalog.sqlite`.
