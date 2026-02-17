# Multimodal Product Recommendation System (Thesis)

Cross-merchant product recommendation engine using CLIP multimodal embeddings.
Covers two verticals: **Arcaplanet** (pet supplies) and **Twinset** (fashion).

## Project Status

| Step | Status | Description |
|------|--------|-------------|
| 1. Scraping pipeline | Done | Hardened scrapers for Arcaplanet & Twinset with session reuse, logging, image download |
| 2. Embedding pipeline | Done | OpenCLIP ViT-B-32 multimodal embeddings + FAISS index |
| 3. Recommendation API | Next | REST API serving similarity-based recommendations |
| 4. Evaluation | Planned | Offline metrics (precision@k, nDCG) and qualitative analysis |

## Catalog

- **647 products** (500 Arcaplanet, 147 Twinset)
- **~2,777 images** downloaded locally
- SQLite database at `data/catalog.sqlite`

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Run scrapers

```bash
python -m scrapers.arcaplanet --max-sitemaps 2 --limit 500 --download-images
python -m scrapers.twinset --max-pages 5 --limit 500 --download-images
```

Artifacts:
- `data/catalog.sqlite`
- `data/images/<merchant>/<product_id>/<idx>.jpg`

## 2. Build embedding index

```bash
python -m pipelines.embed -v
```

Options:
- `--force` — re-embed all products (ignores existing index)
- `--device cpu|cuda|mps` — override device auto-detection
- `--batch-size 32` — image batch size for CLIP encoder

Artifacts:
- `data/catalog.faiss` — FAISS inner-product index (~634 vectors x 512 dims)
- `data/catalog_meta.parquet` — metadata (product_id, merchant, name, price, currency, url)

**How it works:** Each product gets a fused embedding = average of (mean CLIP image embedding, CLIP text embedding of `"{name}. {description[:200]}"`), L2-normalized. Products with no valid name or zero images are filtered out (~13 Twinset products).

## 3. Search the index (verification)

```bash
python -m pipelines.search --query "cibo per gatti" --top-k 5
python -m pipelines.search --query "vestido donna" --top-k 5
python -m pipelines.search --image data/images/twinset/.../0.jpg --top-k 5
```

## Inspect scraped data

```bash
python scripts/summary.py
```

## View SQLite in a GUI

Install **DB Browser for SQLite** (free) and open `data/catalog.sqlite`.

## Tech Stack

- **Scraping:** requests, BeautifulSoup, extruct (JSON-LD)
- **Embeddings:** OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), 512-dim shared image-text space
- **Index:** FAISS `IndexFlatIP` (brute-force cosine similarity on normalized vectors)
- **Storage:** SQLite (catalog), Parquet (metadata), local filesystem (images)
