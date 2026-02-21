# Methodology

This chapter describes the research approach, tools, data collection process, model selection, and system architecture for the cross-merchant multimodal product recommendation system.

## 1. Research Approach

The project builds a **hybrid recommendation system** that combines content-based filtering (multimodal CLIP embeddings) with store-based collaborative filtering, operating across multiple merchants. A conversational AI layer (GPT-4o-mini with tool calling) handles preference elicitation and result presentation, enabling a natural-language shopping assistant experience.

The core hypothesis is that CLIP multimodal embeddings, trained on large-scale web data, can capture both visual and semantic similarity across product categories and languages, enabling effective cross-merchant recommendations. A secondary hypothesis is that store-level collaborative filtering can overcome extreme data sparsity in transaction records to provide meaningful personalized recommendations.

## 2. Preliminary Experiments

### 2.1 Cosine Similarity Validation (Google Colab)

Before building the full pipeline, a preliminary experiment verified that CLIP ViT-B-32 embeddings can meaningfully match text queries to product images.

**Test setup:** Encoded the text query "black converse high top sneaker with white sole" and computed cosine similarity against four product images.

**Results:**

| Image | Cosine Similarity |
|-------|------------------|
| Black Converse high-top | 0.3315 |
| Red Converse high-top | 0.2729 |
| Black generic shoe | 0.2001 |
| Red high heels | -0.0041 |

**Conclusion:** The model correctly ranked the exact match highest, followed by visually similar items, and assigned near-zero similarity to an unrelated product. This confirmed that multimodal embeddings capture both visual features (color, shape) and semantic concepts (brand, style), validating the approach before investing in the full pipeline.

### 2.2 Prototype Web Scraping

Initial scraping tests were conducted on both target websites to validate data extraction feasibility — verifying that product names, descriptions, prices, and images could be reliably extracted from each site's HTML structure and JSON-LD metadata.

## 3. Data Collection

### 3.1 Scraping Pipeline

Two custom scrapers were built, one per merchant:

- **Arcaplanet** — sitemap-driven scraper that discovers product URLs from XML sitemaps, then extracts structured data (JSON-LD, Open Graph, HTML parsing). 500 products scraped.
- **Twinset** — category pagination scraper that traverses category listing pages, extracting product data from each page. 147 products scraped.

Both scrapers share common infrastructure:
- **Session reuse** — persistent HTTP sessions with retry logic (tenacity) to handle transient failures
- **Logging** — structured logging for scrape progress and errors
- **Image download** — all product images downloaded to local filesystem (`data/images/<merchant>/<product_id>/<idx>.jpg`)
- **URL filtering** — deduplication and validation of product URLs before scraping

### 3.2 Storage Schema

Products are stored in a SQLite database (`data/catalog.sqlite`) with the following core tables:

```sql
CREATE TABLE products (
    product_id  TEXT PRIMARY KEY,     -- SHA-256[:24] of "merchant::url"
    merchant    TEXT NOT NULL,        -- "arcaplanet" or "twinset"
    url         TEXT NOT NULL,
    name        TEXT,
    description TEXT,
    price       TEXT,
    currency    TEXT,
    scraped_at  TEXT NOT NULL         -- ISO UTC timestamp
);

CREATE TABLE product_images (
    product_id  TEXT NOT NULL,
    image_url   TEXT NOT NULL,        -- original source URL
    local_path  TEXT NOT NULL,        -- relative path to downloaded file
    PRIMARY KEY (product_id, image_url)
);
```

The `product_id` is a deterministic hash of `"merchant::url"`, ensuring idempotent re-scraping.

### 3.3 Data Quality

| Merchant | Products Scraped | Images Downloaded |
|----------|-----------------|-------------------|
| Arcaplanet | 500 | ~2,040 |
| Twinset | 147 | ~736 |
| **Total** | **647** | **~2,776** |

After filtering products with empty names or zero images, **634 valid products** remained for embedding. The 13 filtered products were primarily Twinset items with missing metadata.

## 4. Transaction Data Analysis

### 4.1 Data Source

Tax-free transaction data was extracted from a Databricks pipeline processing invoice records. The data pipeline (`scripts/azure-sql-ingestion`) is data-agnostic — it processes whatever invoice tables are pointed at in the source configuration, performing:

1. **Schema extraction** — parsing serialized JSON invoice items into structured columns (item code, description, quantity, unit price, VAT rate)
2. **Header join** — linking item-level records with invoice headers containing store, merchant, and customer metadata
3. **Text normalization** — lowercasing, punctuation removal, whitespace normalization, unit token extraction (kg, g, ml, etc.)
4. **Product ID generation** — SHA-256 hash of normalized description text, creating stable product identifiers from free-text descriptions
5. **Store profile aggregation** — computing per-store statistics (revenue, invoice count, product diversity, median price)

### 4.2 Data Volume

The January 2025 test dataset contains:

| Metric | Count |
|--------|-------|
| Transaction items | 24,665 |
| Invoices | 3,865 |
| Distinct customers | 2,951 |
| Distinct merchants | 286 |
| Distinct stores | 345 |
| Distinct products (by normalized description) | 10,893 |

### 4.3 Data Quality Assessment

**Description quality varies dramatically by merchant type:**

- **Fashion items** use extremely generic descriptions: "maglia" (knitwear) appears 357 times, "pantalone" (trousers) 270 times, "giacca" (jacket) 109 times. These descriptions carry almost no discriminative power — a "maglia" at a luxury boutique is indistinguishable from one at a discount store based on description alone.
- **Grocery items** are somewhat more specific: "prosciutto cotto t a gr 100" (cooked ham 100g) or "birra beck s cl.33x5 +1" (Beck's beer 33cl 5+1 pack). Weight/volume tokens add some differentiation.
- **Non-food items** occasionally include brand names: "kindle 7 paperwhite 16gb 2024" provides strong product identity.

**Extreme sparsity characterizes the transaction data:**

| Metric | Value |
|--------|-------|
| Customers with exactly 1 transaction | 79.5% |
| Products appearing exactly once | 65.0% |
| Average items per invoice | 6.4 |
| Median items per invoice | ~4 |

This level of sparsity makes traditional product-level collaborative filtering impractical — most customer-product pairs are unobserved, and the product descriptions are too generic to create meaningful item embeddings from transaction data alone.

**Zero overlap with scraped catalog:** Transaction product IDs (SHA-256 of normalized description) share no overlap with scraped catalog product IDs (SHA-256 of merchant + URL). This is expected — the transaction data covers 286 merchants while the scraped catalog covers only 2, and the ID generation methods differ fundamentally.

## 5. Embedding Pipeline

### 5.1 Model Selection: OpenCLIP ViT-B-32

**Model:** `ViT-B-32` with pretrained weights `laion2b_s34b_b79k` from the OpenCLIP library.

**Rationale:**
- **512-dimensional** shared image-text embedding space — compact enough for brute-force search at this catalog size
- **LAION-2B training data** — 2 billion image-text pairs from the web, providing broad visual and semantic coverage including multilingual content (Italian, English, Spanish, and more)
- **Shared embedding space** — text and image embeddings are directly comparable via cosine similarity, enabling cross-modal search
- **ViT-B-32** offers a good balance between embedding quality and inference speed; larger variants (ViT-L-14) were considered but unnecessary given the catalog size

### 5.2 Text Representation

Each product's text embedding is computed from:

```
"{name}. {description[:200]}"
```

This combines the product identity (name) with descriptive features (first 200 characters of description), truncated to stay within CLIP's 77-token context window while capturing the most informative content.

### 5.3 Image Encoding

All product images are encoded individually through the CLIP visual encoder:
1. Each image is preprocessed using the model's standard transforms (resize, center crop, normalization)
2. Encoded to a 512-dimensional vector
3. L2-normalized per image

For products with multiple images, embeddings are **mean-pooled** (averaged) to produce a single representative image embedding per product.

**Why mean over median:** Mean pooling was chosen because:
- It preserves contributions from all views (front, back, detail shots), giving a more complete visual representation
- Median would discard outlier views that may contain distinctive features
- For approximately normal distributions of image views, mean and median converge; mean is more computationally straightforward

### 5.4 Fusion Strategy

The final product embedding combines visual and textual representations:

```
fused = (mean_image_embedding + text_embedding) / 2
fused = fused / ||fused||₂
```

A 50/50 average gives equal weight to visual and textual signals, then L2-normalization ensures all product vectors lie on the unit hypersphere, making dot product equivalent to cosine similarity.

### 5.5 Technical Challenges

**FAISS/OpenMP Deadlock on macOS:** The initial implementation used FAISS for approximate nearest neighbor search. However, FAISS's OpenMP threading conflicts with PyTorch's thread pool on macOS, causing a deadlock during model loading. The solution was to replace FAISS with numpy brute-force cosine similarity (`matrix @ query.T`), which is performant at this catalog size (634 products × 512 dimensions) and avoids the threading issue entirely.

**MPS (Apple Metal) Acceleration:** On Apple Silicon Macs, the embedding pipeline uses Metal Performance Shaders (MPS) via PyTorch's `mps` device, achieving approximately 4x speedup over CPU for CLIP inference. Device selection is automatic with manual override available.

## 6. Similarity Search

Search is implemented as brute-force cosine similarity via numpy dot product:

```python
scores = matrix @ query_embedding.T  # (N,) cosine similarities
top_indices = np.argsort(scores)[::-1][:top_k]
```

Since all embeddings are L2-normalized, the dot product equals cosine similarity. This approach is:
- **Exact** — no approximation error (unlike ANN methods)
- **Fast enough** — sub-millisecond for 634 × 512 matrix multiplication
- **Simple** — no additional index structures or libraries needed

For larger catalogs (10k+ products), approximate methods (FAISS, ScaNN) would be warranted.

## 7. Store-Based Collaborative Filtering

### 7.1 Rationale

Given the extreme sparsity of the transaction data (79.5% of customers have only 1 transaction, 65% of products appear only once), traditional product-level collaborative filtering is impractical. Store-level aggregation solves this problem by:

- **Smoothing individual product sparsity** — a store's product mix (e.g., 31 distinct products across 52 invoices) creates a denser, more informative signal than any single product
- **Capturing store "personality"** — even with generic descriptions like "maglia" and "pantalone", the mix of products at a store reveals its character (fashion boutique vs. grocery vs. electronics)
- **Working with available granularity** — the 345 stores with aggregated profiles provide sufficient density for meaningful similarity computation

### 7.2 Item-Based CF (Store Similarity)

**Approach:** Build a store-product interaction matrix where rows are stores, columns are product IDs, and values are total quantity sold. Compute pairwise cosine similarity between store vectors.

```
Matrix dimensions: 345 stores × 10,893 products (sparse)
Similarity: cosine_similarity(store_A, store_B)
```

**Interpretation:** Two stores are similar if they sell similar products in similar proportions. This captures store-level patterns that transcend individual product descriptions — a store selling "maglia", "pantalone", "giacca", and "cappotto" has a clear fashion profile even though each description is generic.

**Use case:** "Stores like this one" — given a store, find other stores with the most similar product mix. Useful for market analysis and cross-merchant discovery.

### 7.3 User-Based CF (Co-Shopping Patterns)

**Approach:** Build a customer-store matrix where rows are customers, columns are stores, and values are total spend. For a given customer, find similar customers via cosine similarity on store-visit vectors, then recommend stores that similar customers visited but the target customer hasn't.

```
Matrix dimensions: 2,951 customers × 345 stores (sparse)
Steps: 1) Find top-20 similar customers
        2) Aggregate their unvisited stores weighted by similarity
        3) Return top-k ranked stores
```

**Interpretation:** Customers who shop at similar stores likely have similar preferences. If customer A shops at stores X, Y, Z and similar customer B shops at X, Y, W, then W is a candidate recommendation for customer A.

**Limitations:** With 79.5% of customers having only 1 transaction, most customer vectors are very sparse (single non-zero entry). The model works best for the ~20% of customers with repeat visits.

### 7.4 Mock Profiles for Demonstration

Three mock profiles map to real customer IDs from the transaction data, each representing a distinct shopping pattern:

| Profile | Shopping Pattern | Merchants | Invoices |
|---------|-----------------|-----------|----------|
| Luca | Fashion enthusiast | Beat, FMC, Conceptlab | 6 |
| Sofia | Grocery regular | Granmercato, area2rezzonico | 5 |
| Maria | Versatile shopper | Bordoni, Peter Pan, Granmercato | 6 |

A Guest profile (no purchase history) demonstrates the system's fallback to content-based recommendations only.

## 8. Favorites & User Interaction Data

### 8.1 Purpose

The favorites feature serves as a lightweight implicit feedback collection mechanism. Users can "heart" products to save them, generating preference signals that could be used for:

- **Future model training** — favorited products provide positive labels for learning user preferences
- **Session-based personalization** — favorites persist across page navigations and browser sessions
- **Evaluation data** — comparing favorited products against recommendations provides a proxy for recommendation quality

### 8.2 Implementation

Favorites are session-based using UUIDs stored in localStorage, requiring no authentication:

```sql
CREATE TABLE favorites (
    session_id  TEXT NOT NULL,
    product_id  TEXT NOT NULL,
    added_at    TEXT NOT NULL,
    PRIMARY KEY (session_id, product_id)
);
```

When a user profile is selected, the `session_id` maps to the `profile_id`, linking favorites to the profile's purchase history. Guest users receive a random UUID.

## 9. Recommendation System Architecture

### 9.1 Three-Tier Approach

The system combines three complementary recommendation strategies:

1. **Content-based (CLIP embeddings)** — finds products similar by visual appearance and textual description. This is the primary recommendation engine, powering the chatbot's product search. Works for all users regardless of purchase history (no cold-start problem).

2. **Store-based collaborative filtering** — finds stores with similar product mixes (item-based) or stores visited by similar customers (user-based). This layer operates at the store level to overcome product-level sparsity. Results are displayed on a dedicated Evaluation page, separate from the chatbot.

3. **Conversational AI (GPT-4o-mini)** — mediates between the user and the search engine, performing preference elicitation, query formulation, intelligent result selection, and natural language presentation.

### 9.2 Content-Based Layer (Chatbot)

The conversational AI layer uses GPT-4o-mini with tool calling:

1. **Preference elicitation** — the LLM asks clarifying questions about budget, brand, category, and specific needs
2. **Query formulation** — the LLM translates user preferences into structured search parameters (text query, price range, merchant filter)
3. **Intelligent selection** — after search returns candidates, the LLM calls `select_products` to choose only genuinely relevant matches for the carousel (solving the problem of always showing a fixed number of results)
4. **Store discovery** — the LLM calls `find_nearby_stores` to show physical store locations alongside product recommendations
5. **Iterative refinement** — users can refine ("show me something cheaper") and the LLM adjusts the search

**Quality control:** A minimum similarity score threshold (0.10) filters out poor matches before the LLM sees them. The LLM then applies its own judgment about which remaining products to recommend, ensuring the carousel contains only confident recommendations.

**Tool calling interface:**

| Tool | Purpose |
|------|---------|
| `search_products` | Text-based CLIP search with optional price/merchant filters |
| `search_by_image` | Image-based CLIP search for visual similarity |
| `select_products` | LLM-curated selection of which search results to display |
| `ask_preferences` | Present follow-up questions to the user |
| `explain_recommendation` | Generate plain-language explanation of why a product was recommended |
| `find_nearby_stores` | Locate physical stores for a merchant, with optional distance sorting |

### 9.3 Collaborative Filtering Layer (Eval Page)

The CF layer is intentionally separated from the chatbot to maintain clear boundaries between recommendation approaches:

- **Item-based CF** — for a given profile's most-visited store, shows stores with the most similar product mix (cosine similarity on store-product vectors)
- **User-based CF** — for a given profile's customer, shows stores visited by similar customers but not yet visited by this customer

Results are displayed on the Eval page with profile selection, enabling side-by-side comparison of how different user profiles receive different store recommendations.

### 9.4 Explainability

When a user asks "why did you recommend this?", the system provides the LLM with:
- The original search query
- The product's similarity score
- The product's price and merchant
- Context about what the user was looking for

The LLM then generates a user-friendly explanation (e.g., "This was recommended because it closely matches your request for affordable cat food — it's within your budget at 9.29 EUR and is highly similar to what you described."). This approach provides transparency without exposing technical details like embedding distances.

## 10. API & Frontend Design

### 10.1 FastAPI REST API

The backend is a FastAPI application that:
- Loads the CLIP model, embedding index, and CF matrices once at startup (via lifespan context manager)
- Maintains a persistent SQLite connection for metadata lookups
- Serves product images as static files
- Provides REST endpoints for search, browsing, chatbot interaction, favorites, and collaborative filtering

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/products` | Paginated catalog browsing |
| GET | `/api/products/{id}` | Product detail with all images |
| GET | `/api/products/{id}/similar` | "More like this" recommendations |
| GET | `/api/search/text?q=...` | CLIP text search |
| POST | `/api/search/image` | CLIP image search |
| POST | `/api/chat` | Conversational chatbot (GPT-4o-mini + tool calling) |
| GET | `/api/stores` | Store locations |
| GET/POST/DELETE | `/api/favorites` | Manage user favorites |
| GET | `/api/stores/{id}/similar` | Item-based CF: similar stores |
| GET | `/api/recommend/stores` | User-based CF: store recommendations |
| GET | `/api/profiles` | Available user profiles |

The chat endpoint keeps the OpenAI API key server-side and orchestrates the tool-calling loop, preventing API key exposure in the browser.

### 10.2 React Frontend

The frontend is a React 18 single-page application built with Vite and Tailwind CSS:

- **ChatPage** — conversational interface with text input, image upload, inline product carousels, and store location maps. Always mounted to preserve conversation state across page navigation.
- **BrowsePage** — paginated product grid with merchant filtering and favorites toggle
- **MapPage** — interactive Leaflet map with store markers and merchant filtering
- **EvalPage** — collaborative filtering dashboard with profile selection, showing side-by-side item-based and user-based store recommendations

Key components:
- `ProductCard` — displays product image, name, price, merchant badge, similarity score, external link, and heart button for favorites
- `ProductCarousel` — horizontal scrollable row of product cards, rendered inline in chat messages
- `StoreMap` — inline Leaflet map rendered alongside product carousels in chat responses
- `ImageUpload` — file picker for image-based search
- `ChatMessage` — chat bubble supporting text, images, product carousels, store maps, and clickable follow-up question chips
- `ProfilePicker` — profile selection cards for the collaborative filtering demo

## 11. Multilingual Capabilities

The system supports multilingual queries without explicit language detection or translation:

- **LAION-2B training data** includes multilingual web content (image-text pairs in many languages)
- The CLIP model implicitly learns cross-lingual associations through shared visual grounding
- Tested successfully with Italian ("cibo per gatti"), English ("cat food"), and Spanish ("vestido elegante") queries

This is a property of the training data distribution rather than an explicit multilingual architecture — performance may vary for underrepresented languages.

## 12. Evaluation Methodology

### 12.1 Models Evaluated

The system evaluates five recommendation models to compare approaches and validate against baselines:

| Model | Level | Method |
|-------|-------|--------|
| Content-Based (CLIP) | Product | Self-retrieval: use each product's embedding as query, relevance = same merchant |
| Item-Based CF | Store | Leave-one-out: hold out one store per customer with 2+ visits |
| User-Based CF | Store | Leave-one-out: same setup, using customer-store interaction matrix |
| Random Baseline | Both | Uniform random sample of k items/stores |
| Popularity Baseline | Both | Most-visited stores / most-common products |

### 12.2 Metrics

Six standard information retrieval metrics are computed for each model:

- **Precision@K** — fraction of top-k recommendations that are relevant. Measures recommendation accuracy.
- **Recall@K** — fraction of all relevant items that appear in the top-k list. Measures completeness.
- **nDCG@K** — normalized discounted cumulative gain. Measures ranking quality by penalizing relevant items appearing at lower positions.
- **Hit Rate@K** — binary metric: 1 if any relevant item appears in top-k, else 0. Measures whether the system finds at least one relevant result.
- **Coverage** — fraction of the full catalog that appears in any recommendation list across all test cases. Measures how much of the catalog the model explores.
- **Diversity** — average pairwise dissimilarity among recommended items (1 - cosine similarity). Measures variety in recommendations.

### 12.3 Evaluation Protocol

**Product-level (Content-Based + baselines):** For each of the 634 products, use its embedding as query, exclude itself from results, and check if same-merchant products rank in the top-k. This self-retrieval protocol tests whether the model groups products by merchant correctly — a necessary condition for meaningful cross-product recommendations.

**Store-level (CF + baselines):** Leave-one-out cross-validation. For each customer with 2+ distinct store visits, hold out one store, generate recommendations from the remaining visits, and check whether the held-out store appears in the top-k recommendations. This directly tests the CF models' ability to predict future store visits.

### 12.4 Baseline Definitions

**Random Baseline:** For each test case, uniformly sample k items/stores from the candidate set (excluding already-visited items). Provides a lower bound — any useful model should substantially outperform random selection.

**Popularity Baseline:** Always recommend the most-visited stores (store-level) or products from the largest merchant group (product-level). Tests whether the models learn anything beyond simple frequency patterns.

### 12.5 Implementation

Evaluation is implemented in `pipelines/evaluation.py` and exposed via `GET /api/evaluation/results?k=5` (supports k=3, 5, 10). Results are cached in `app.state._eval_cache` to avoid recomputation. All evaluations work on matrix copies to avoid corrupting live CF matrices.

## 13. Future Work

- **Catalog expansion** — additional merchants and product categories to test cross-domain recommendations at scale
- **Embedding model upgrades** — larger CLIP variants (ViT-L-14) or domain-specific fine-tuning if quality improvements are needed
- **Real-time feedback loop** — using favorites and click data to personalize content-based recommendations over time
- **Hybrid ranking** — combining content-based similarity scores with collaborative filtering signals into a unified recommendation score
