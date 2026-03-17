# Multimodal Tourist Shopping Recommendation System

Cross-merchant recommendation engine for tax-free tourist shopping near Como/Milan.
Combines CLIP multimodal embeddings (product level) with store-level collaborative filtering and demographic-aware models.

**Merchants:** Arcaplanet (500 pet-supply products) · Twinset (134 fashion products)
**Customers:** ~40k · **Transactions:** ~80k · **Stores:** 776

---

## Environment Setup

### Python — requires 3.12 (LightFM is incompatible with 3.13)

```bash
conda create -n thesis-reco python=3.12 -y
conda activate thesis-reco
pip install -r requirements.txt
```

> Always activate `thesis-reco` before running any Python command.
> All commands below assume the project root (`thesis-reco/`) as working directory.

### Node.js — requires 20+

```bash
nvm install 20
nvm use 20          # add `nvm alias default 20` to make it permanent
cd frontend && npm install
```

### OpenAI API key (for chatbot)

```bash
export OPENAI_API_KEY=sk-...
```

---

## One-Time Data Setup

Run once after cloning to populate the customer demographics table:

```bash
python -m scripts.populate_customer_demographics
```

This joins `data/customer_info.xlsx` (date of birth) with `data/final_transaction_data.csv` (nationality, residency) and writes a `customer_demographics` table to `data/catalog.sqlite`.
Output: 39,777 records — tourist types: cross_border=28,855 · international=10,463 · domestic=459.

---

## Running the Application

Two terminals, both from the **project root**:

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload

# Terminal 2 — React frontend (from frontend/ subdirectory)
cd frontend && npm run dev
```

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

---

## Running the Evaluation

Generates `data/evaluation_results.json` (used by the `/eval` dashboard).
**Must re-run whenever model code or evaluation logic changes.**

```bash
python -m scripts.run_evaluation
```

This evaluates all 13 models at k = 3, 5, 10, 15, 20 and takes ~10–15 minutes
(LightFM training is the bottleneck). Progress is printed for each model.

### What it evaluates

| # | Model | Level | Paradigm |
|---|-------|-------|----------|
| 1 | Content-Based (CLIP) | Product | Self-retrieval, relevance = same category |
| 2 | Item-Based CF | Store | Memory-based, temporal split |
| 3 | User-Based CF | Store | Memory-based, temporal split |
| 4 | ALS | Store | Model-based, temporal split |
| 5 | LightFM WARP | Store | Model-based, temporal split |
| 6 | LightFM Hybrid | Store | LightFM + store/user side features |
| 7 | Random Baseline | Store | Baseline |
| 8 | Popularity Baseline | Store | Baseline |
| 9 | Random Baseline (Product) | Product | Baseline |
| 10 | Popularity Baseline (Product) | Product | Baseline |
| 11 | Demographic Popularity | Store | Segment-based popularity |
| 12 | LightFM Demo | Store | LightFM + demographic user features |
| 13 | LightFM Full Hybrid | Store | LightFM + behavioral + demographic features |

### Metrics

| Metric | Applies to |
|--------|-----------|
| Hit Rate@K | Store + Product |
| nDCG@K | Store + Product |
| Precision@K | Store + Product |
| Recall@K | Store + Product |
| Coverage | Store + Product |
| Diversity (ILD) | Store + Product |
| Novelty | Product only |

**Product-level relevance:** same product category (Twinset: URL slug → first word; Arcaplanet: name keyword matching for animal × product type). Coverage: 561/634 products (88.5%). Remaining 11.5% fall back to same-merchant.

**Store-level protocol:** temporal train/test split at 2025-12-01. Test cases = customers who visited at least one new store after the split date (1,708 cases).

### Hyperparameter comparison (optional)

```bash
python -m scripts.run_hyperparameter_comparison
```

Outputs `data/hyperparameter_results.json` with grid-search results for ALS and LightFM WARP.

---

## Other Useful Commands

### Rebuild the CLIP embedding index

Only needed if new products are added to the database:

```bash
python -m pipelines.embed                  # incremental (new products only)
python -m pipelines.embed --force          # rebuild everything
python -m pipelines.embed --device cpu     # force CPU (default: MPS on Apple Silicon)
```

Artifacts: `data/catalog_embeddings.npy` (634 × 512) · `data/catalog_meta.parquet`

### Test the search index (CLI)

```bash
python -m pipelines.search --query "cibo per gatti" --top-k 5
python -m pipelines.search --query "vestido elegante" --top-k 5
```

### Kill stale dev servers

```bash
lsof -ti :8000,:5173,:5174 | xargs kill -9
```

---

## Project Structure

```
thesis-reco/
├── api/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── chat.py              # GPT-4o-mini tool-calling logic
│   └── models.py            # Pydantic request/response schemas
├── pipelines/
│   ├── embed.py             # CLIP embedding pipeline
│   ├── search.py            # Index loading + cosine search
│   ├── collab_model.py      # CF models: ALS, LightFM, feature builders
│   ├── demographic.py       # Demographic models + feature builders
│   └── evaluation.py        # All evaluation logic + metrics
├── scripts/
│   ├── run_evaluation.py                  # Full evaluation runner
│   ├── run_hyperparameter_comparison.py   # Grid search for ALS/LightFM
│   └── populate_customer_demographics.py  # One-time demographics setup
├── frontend/src/
│   ├── pages/
│   │   ├── LandingPage.jsx      # Profile picker
│   │   ├── RecommenderPage.jsx  # CF + chatbot
│   │   ├── BrowsePage.jsx       # Product catalog
│   │   ├── MapPage.jsx          # Store map
│   │   └── EvalPage.jsx         # Evaluation dashboard
│   └── components/
│       ├── CFSection.jsx        # Store CF results (4 columns incl. demographic)
│       ├── ChatSection.jsx      # Chatbot interface
│       └── ...
├── data/
│   ├── catalog.sqlite           # Main DB: products, transactions, stores, demographics
│   ├── catalog_embeddings.npy   # 634 × 512 CLIP embeddings
│   ├── catalog_meta.parquet     # Product metadata
│   ├── evaluation_results.json  # Pre-computed eval results (served by API)
│   └── images/                  # Downloaded product images
└── docs/
    └── methodology.md           # Full methodology documentation
```

---

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Profile picker — must select profile before entering |
| `/recommend` | Recommender | CF store recommendations (4 columns) + chatbot |
| `/browse` | Browse | Paginated product catalog with merchant filter + favorites |
| `/map` | Map | Interactive store map with merchant filter |
| `/eval` | Evaluation | 13-model comparison dashboard, split by level, K selector |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/products` | Paginated catalog (`?merchant=`, `?page=`, `?per_page=`) |
| GET | `/api/products/{id}` | Product detail with images |
| GET | `/api/products/{id}/similar` | CLIP similarity — "more like this" |
| GET | `/api/search/text?q=` | CLIP text search |
| POST | `/api/search/image` | CLIP image search |
| POST | `/api/chat` | GPT-4o-mini conversational chatbot |
| GET | `/api/stores` | Store locations (`?merchant=`) |
| GET | `/api/stores/{id}/similar` | Item-based CF: similar stores |
| GET | `/api/recommend/stores` | User-based CF: store recommendations |
| GET | `/api/recommend/stores/lightfm` | LightFM WARP recommendations |
| GET | `/api/recommend/stores/demographic` | Demographic Popularity recommendations |
| GET | `/api/cities` | Sorted list of all store cities |
| GET | `/api/profiles` | Available user profiles |
| GET/POST/DELETE | `/api/favorites` | Manage session favorites |
| GET | `/api/evaluation/results?k=5` | Pre-computed evaluation results |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Embeddings | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`), 512-dim |
| CF — memory | scipy sparse matrices, sklearn cosine similarity |
| CF — model | `implicit` (ALS), `lightfm` (WARP + hybrid) |
| API | FastAPI, uvicorn |
| Chatbot | GPT-4o-mini with tool calling (OpenAI) |
| Frontend | React 18, Vite, Tailwind CSS, Leaflet |
| Storage | SQLite, Parquet, NumPy npy, local filesystem |
| Python | 3.12 (conda env `thesis-reco`) |
| Node.js | 20+ (via nvm) |
