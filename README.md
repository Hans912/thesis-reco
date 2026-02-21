# Multimodal Product Recommendation System (Thesis)

Cross-merchant product recommendation engine using CLIP multimodal embeddings and a conversational AI chatbot.
Covers two verticals: **Arcaplanet** (pet supplies) and **Twinset** (fashion).

## Project Status

| Step | Status | Description |
|------|--------|-------------|
| 1. Scraping pipeline | Done | Hardened scrapers for Arcaplanet & Twinset with session reuse, logging, image download |
| 2. Embedding pipeline | Done | OpenCLIP ViT-B-32 multimodal embeddings + numpy cosine search |
| 3. Recommendation API | Done | FastAPI REST API with CLIP search endpoints |
| 4. Chatbot frontend | Done | React + GPT-4o-mini conversational recommender with tool calling |
| 5. Evaluation | Done | Offline metrics (Precision@K, Recall@K, nDCG@K, Hit Rate@K, Coverage, Diversity) across 5 models |

## Pages

| Route | Page | Content |
|-------|------|---------|
| `/` | Landing | Full-screen profile picker — select a profile or Guest before entering |
| `/recommend` | Recommender | CF store recommendations (top) + chatbot product search (bottom) |
| `/browse` | Browse | Paginated product catalog with merchant filter and favorites |
| `/map` | Map | Interactive Leaflet map of store locations |
| `/eval` | Evaluation | Model comparison dashboard — 5 models x 6 metrics with bar charts |

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

For the frontend (requires Node.js 20+):

```bash
cd frontend
npm install
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

## 3. Run the application

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

Start the API and frontend in two separate terminals:

```bash
# Terminal 1 — API server (loads CLIP model, serves recommendations + chatbot)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — React frontend
cd frontend && npm run dev
```

- Frontend: `http://localhost:5173`
- API docs (Swagger UI): `http://localhost:8000/docs`

**API endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/products` | Browse catalog (paginated, optional `?merchant=` filter) |
| GET | `/api/products/{id}` | Product detail with all images |
| GET | `/api/products/{id}/similar` | "More like this" recommendations |
| GET | `/api/search/text?q=...` | Text search via CLIP |
| POST | `/api/search/image` | Image upload search via CLIP |
| POST | `/api/chat` | Conversational chatbot (GPT-4o-mini + tool calling) |
| GET | `/api/stores` | Store locations (optional `?merchant=` filter) |
| GET | `/api/stores/{id}/similar` | Item-based CF: similar stores |
| GET | `/api/recommend/stores` | User-based CF: store recommendations |
| GET | `/api/profiles` | Available user profiles |
| GET | `/api/evaluation/results?k=5` | Offline evaluation results (k=3, 5, or 10) |

## 4. Search the index (CLI verification)

```bash
python -m pipelines.search --query "cibo per gatti" --top-k 5
python -m pipelines.search --query "vestido donna" --top-k 5
python -m pipelines.search --image data/images/twinset/.../0.jpg --top-k 5
```

## Evaluation

The evaluation dashboard (`/eval`) compares 5 recommendation models:

| Model | Method |
|-------|--------|
| Content-Based (CLIP) | Self-retrieval with same-merchant relevance |
| Item-Based CF | Leave-one-out on store-product similarity |
| User-Based CF | Leave-one-out on customer-store patterns |
| Random Baseline | Uniform random selection |
| Popularity Baseline | Most-visited stores / largest merchant group |

Metrics computed: Precision@K, Recall@K, nDCG@K, Hit Rate@K, Coverage, Diversity.

Results are cached server-side after first computation.

## Tech Stack

- **Scraping:** requests, BeautifulSoup, extruct (JSON-LD)
- **Embeddings:** OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), 512-dim shared image-text space
- **Search:** numpy brute-force cosine similarity (sufficient for ~634 products)
- **API:** FastAPI (REST), uvicorn (ASGI server)
- **Chatbot:** GPT-4o-mini with tool calling (OpenAI API)
- **Frontend:** React 18, Vite, Tailwind CSS
- **Storage:** SQLite (catalog), Parquet (metadata), npy (embeddings), local filesystem (images)

## Known Issues

- `faiss-cpu` causes a deadlock with PyTorch on macOS (OpenMP threading conflict). Use numpy-based search instead — performance is identical at this catalog size.
