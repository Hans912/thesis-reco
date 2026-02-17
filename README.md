# Multimodal Product Recommendation System (Thesis)

Cross-merchant product recommendation engine using CLIP multimodal embeddings.
Covers two verticals: **Arcaplanet** (pet supplies) and **Twinset** (fashion).

## Project Status

| Step | Status | Description |
|------|--------|-------------|
| 1. Scraping pipeline | Done | Hardened scrapers for Arcaplanet & Twinset with session reuse, logging, image download |
| 2. Embedding pipeline | Done | OpenCLIP ViT-B-32 multimodal embeddings + numpy cosine search |
| 3. Recommendation API | Next | REST API serving similarity-based recommendations |
| 4. Evaluation | Planned | Offline metrics (precision@k, nDCG) and qualitative analysis |

## Catalog

- **647 products** (500 Arcaplanet, 147 Twinset), **634 valid** (after filtering empty names/zero images)
- **~2,776 images** downloaded locally
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
- `data/catalog.sqlite` — product catalog (SQLite)
- `data/images/<merchant>/<product_id>/<idx>.jpg` — product images

## 2. Build embedding index

```bash
python -m pipelines.embed
```

Options:
- `--force` — re-embed all products (ignores existing index)
- `--device cpu|cuda|mps` — override device auto-detection (default: MPS on Apple Silicon, else CPU)
- `--batch-size 32` — batch size for CLIP encoder

Artifacts:
- `data/catalog_embeddings.npy` — embedding matrix (634 x 512, ~1.3 MB)
- `data/catalog_meta.parquet` — product metadata (product_id, merchant, name, price, currency, url)

**How it works:** Each product gets a fused embedding = average of (mean CLIP image embedding, CLIP text embedding of `"{name}. {description[:200]}"`), L2-normalized. Products with no valid name or zero images are filtered out (~13 Twinset products). Similarity search uses numpy dot product on normalized vectors (equivalent to cosine similarity).

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
- **Search:** numpy brute-force cosine similarity (sufficient for ~634 products)
- **Storage:** SQLite (catalog), Parquet (metadata), npy (embeddings), local filesystem (images)

## Known Issues

- `faiss-cpu` causes a deadlock with PyTorch on macOS (OpenMP threading conflict). Use numpy-based search instead — performance is identical at this catalog size.
