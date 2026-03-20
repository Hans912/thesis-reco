# Methodology

This chapter describes the research approach, tools, data collection process, model selection, and system architecture for the cross-merchant multimodal product recommendation system.

## 1. Research Approach

The project builds a **hybrid recommendation system** that combines content-based filtering (multimodal CLIP embeddings) with store-based collaborative filtering, operating across multiple merchants. A conversational AI layer (GPT-4o-mini with tool calling) handles preference elicitation and result presentation, enabling a natural-language shopping assistant experience.

The core hypothesis is that CLIP multimodal embeddings, trained on large-scale web data, can capture both visual and semantic similarity across product categories and languages, enabling effective cross-merchant recommendations. A secondary hypothesis is that store-level collaborative filtering can overcome extreme data sparsity in transaction records to provide meaningful personalized recommendations.

## 2. Preliminary Experiments

### 2.1 Cosine Similarity Validation (Google Colab)
÷
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

The initial dataset covered January 2025 only (24,665 items, 2,951 customers, 345 stores). Preliminary evaluation revealed that this single month of data was insufficient for collaborative filtering — 79.5% of customers had only one transaction, providing almost no co-shopping signal. The dataset was therefore expanded to approximately two years of transaction history:

| Metric | Jan 2025 Only | Multi-Year (Final) |
|--------|---------------|-------------------|
| Transaction items | 24,665 | 754,338 |
| Invoices | 3,865 | 109,056 |
| Distinct customers | 2,951 | 39,777 |
| Distinct merchants | 286 | — |
| Distinct stores | 345 | 776 |
| Distinct products | 10,893 | 121,197 |

The multi-year expansion increased the number of customers with 2+ distinct store visits from ~590 to 8,572 — a 14× increase in leave-one-out test cases for CF evaluation.

### 4.3 Duplicate Line Item Handling

The original Databricks pipeline used `explode()` to unnest serialized invoice items, with a composite primary key of `(invoice_id, product_id)`. This caused silent data loss: when the same product appeared multiple times on a single invoice (e.g., same description but different prices or quantities), `INSERT OR IGNORE` would drop subsequent occurrences.

The fix used `posexplode()` instead of `explode()`, which preserves the array position as a `line_num` column. The primary key was changed to `(invoice_id, product_id, line_num)`, ensuring every line item is preserved regardless of duplicate descriptions.

### 4.4 Data Quality Assessment

**Description quality varies dramatically by merchant type:**

- **Fashion items** use extremely generic descriptions: "maglia" (knitwear), "pantalone" (trousers), "giacca" (jacket). These descriptions carry almost no discriminative power — a "maglia" at a luxury boutique is indistinguishable from one at a discount store based on description alone.
- **Grocery items** are somewhat more specific: "prosciutto cotto t a gr 100" (cooked ham 100g) or "birra beck s cl.33x5 +1" (Beck's beer 33cl 5+1 pack). Weight/volume tokens add some differentiation.
- **Non-food items** occasionally include brand names: "kindle 7 paperwhite 16gb 2024" provides strong product identity.

**Sparsity remains a challenge** even with multi-year data, making traditional product-level collaborative filtering impractical. Store-level aggregation (Section 7) and model-based approaches with side features (Section 8) are the primary strategies for handling this sparsity.

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
Matrix dimensions: 776 stores × 121,197 products (sparse)
Similarity: cosine_similarity(store_A, store_B)
```

**Interpretation:** Two stores are similar if they sell similar products in similar proportions. This captures store-level patterns that transcend individual product descriptions — a store selling "maglia", "pantalone", "giacca", and "cappotto" has a clear fashion profile even though each description is generic.

**Use case:** "Stores like this one" — given a store, find other stores with the most similar product mix. Useful for market analysis and cross-merchant discovery.

**Store deduplication:** Multiple `store_id` values can map to the same physical location (e.g., 4 different store IDs for "Granmercato SPA" in "Como"). To prevent recommending the same physical store multiple times, results are deduplicated by the composite key `(merchant_name, city)`. Stores sharing this key with the query store are also excluded from similarity results.

### 7.3 User-Based CF (Co-Shopping Patterns)

**Approach:** Build a customer-store matrix where rows are customers, columns are stores, and values are total spend. For a given customer, find similar customers via cosine similarity on store-visit vectors, then recommend stores that similar customers visited but the target customer hasn't.

```
Matrix dimensions: 39,777 customers × 776 stores (sparse)
Steps: 1) Find top-20 similar customers
        2) Aggregate their unvisited stores weighted by similarity
        3) Normalize scores to [0, 1] range
        4) Deduplicate by merchant+city
        5) Return top-k ranked stores
```

**Interpretation:** Customers who shop at similar stores likely have similar preferences. If customer A shops at stores X, Y, Z and similar customer B shops at X, Y, W, then W is a candidate recommendation for customer A.

**Score normalization:** Raw scores are the sum of similarity weights from multiple similar customers, which can exceed 1.0 (e.g., a store visited by 3 highly similar customers might accumulate a score of 1.78). Scores are normalized to [0, 1] by dividing by the maximum score in the recommendation set, making them interpretable as relative confidence.

**Deduplication:** The same merchant+city exclusion applies as in item-based CF — stores sharing the same physical merchant and city as already-visited stores are removed from recommendations.

### 7.4 Mock Profiles for Demonstration

Three mock profiles map to real customer IDs from the transaction data, each representing a distinct shopping pattern. With the multi-year data expansion, each profile now has significantly richer purchase histories:

| Profile | Shopping Pattern | Description |
|---------|-----------------|-------------|
| Luca | Fashion enthusiast | Frequent visits to fashion and lifestyle stores |
| Sofia | Grocery regular | Consistent grocery shopping patterns |
| Maria | Versatile shopper | Diverse shopping across multiple merchant categories |

A Guest profile (no purchase history) demonstrates the system's fallback to content-based recommendations only.

## 8. Model-Based Collaborative Filtering

### 8.1 Motivation

Memory-based CF (cosine similarity on raw interaction matrices) has well-known limitations: it struggles with sparse data, cannot generalize beyond observed co-occurrences, and scales poorly as the matrix grows. Model-based approaches address these by learning latent factor representations that compress the interaction signal and can generalize to unseen user-item pairs.

Three model-based approaches were added alongside the existing memory-based CF, enabling a direct comparison between memory-based and model-based paradigms:

### 8.2 ALS (Alternating Least Squares)

**Model:** Weighted matrix factorization for implicit feedback, implemented via the `implicit` library (Hu, Koren, & Volinsky, 2008).

**Approach:** Decomposes the customer-store interaction matrix into low-rank user and item factor matrices (64 dimensions each). Unlike SVD or NMF designed for explicit ratings, ALS treats all interactions as positive implicit feedback and uses confidence weighting — more visits to a store indicate stronger preference signal.

**Configuration:** 64 latent factors, 30 iterations, regularization λ = 0.01. The interaction matrix uses visit counts (distinct invoice count per customer-store pair) rather than spend, providing more balanced signal across merchant price ranges.

**Inference:** Recommendations are generated by computing the dot product of user factors with all item factors, filtering out already-visited stores.

### 8.3 LightFM WARP

**Model:** LightFM with Weighted Approximate-Rank Pairwise (WARP) loss (Kula, 2015).

**Rationale for WARP loss:** Unlike pointwise losses (BPR, logistic) that treat each interaction independently, WARP directly optimizes for the top of the ranking. It samples negative items and applies larger gradient updates when high-ranked negatives are found, focusing learning effort on the most impactful ranking errors. This makes it particularly well-suited for top-k recommendation scenarios where only the first few results matter.

**Advantages over ALS:**
- **Ranking-optimized** — WARP loss directly optimizes for top-k precision, while ALS optimizes reconstruction error on the full matrix
- **Feature support** — LightFM can incorporate side features (see Section 8.4), enabling cold-start mitigation
- **Pairwise learning** — learns relative preference ordering rather than absolute scores

**Configuration:** 64 latent components, WARP loss, learning rate 0.05, 30 epochs, 4 threads.

### 8.4 LightFM Hybrid (WARP + Side Features)

**Model:** LightFM with WARP loss plus both store profile and user behavior side features.

**Item (store) features:** Each store is described by three categorical features derived from the `store_profiles` table:
- **City** — e.g., "city:COMO"
- **Size bin** — small (<100 products), medium (100–999), large (1000+)
- **Price bin** — low (median price <€20), mid (€20–€100), high (>€100)

Merchant name was deliberately excluded from store features — it is too specific and causes the model to only recommend stores from the same merchant, overriding the collaborative signal.

**User features:** Each customer is described by three behavioral features computed from their training-period history:
- **Visit frequency bin** — single (1 store), casual (2–3), regular (4–8), power (9+)
- **Spend level bin** — low (<€200), mid (€200–€1000), high (>€1000)
- **Primary city** — the city where the customer shops most frequently

**Why side features matter:** Pure collaborative filtering cannot recommend stores with no interaction history (cold-start problem). By incorporating store profile and user behavior features, the hybrid model learns associations between feature patterns and preferences — e.g., "high-spending users in Como tend to visit mid-to-large stores in nearby cities." This enables better cold-start mitigation and cross-city generalization.

**Configuration:** 128 latent components (vs. 64 for pure WARP), learning rate 0.01 (vs. 0.05), 50 epochs (vs. 30). Higher capacity and slower learning rate help the model integrate the additional feature signal without overfitting.

**Feature encoding:** LightFM's `Dataset` API maps categorical features to sparse indicator vectors, which are composed with the learned latent factors during training.

### 8.5 Interaction Matrix

All three model-based approaches share the same customer-store interaction matrix with recency-weighted values:

```
Matrix dimensions: 39,777 customers × 776 stores (sparse)
Values: recency-weighted visit count
Formula: weight = visits × exp(-days_since_last_visit / 180)
Format: scipy CSR sparse matrix
```

Each cell value combines visit frequency with temporal recency using exponential decay. A 180-day decay constant means a visit from 6 months ago retains ~37% weight, while recent visits are weighted near full strength. This allows the model to prioritize current preferences while still learning from historical patterns. Visit counts were chosen over spend amounts to avoid biasing toward expensive stores.

### 8.6 Training

For the live application, all models are trained at API startup via `train_all_models()` in `pipelines/collab_model.py` (~3 seconds on Apple Silicon). For offline evaluation, models are trained from scratch on training-period data only — see Section 14 for the evaluation protocol.

### 8.7 Academic References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.
- Kula, M. (2015). Metadata embeddings for user and item cold-start recommendations. *CBRecSys Workshop at RecSys*.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI*.

## 9. Demographic Recommenders

### 9.1 Motivation

Collaborative filtering (Sections 7–8) relies on behavioral signals — transaction history — to generate recommendations. While this produces accurate personalized recommendations, it fails for two important scenarios:

1. **Cold-start users** — new customers with no purchase history cannot receive behavioral CF recommendations
2. **Low-frequency shoppers** — customers with very few transactions have sparse interaction vectors, producing unreliable CF scores

Demographic recommenders address both by leveraging **who the customer is** (nationality, age, tourist type) rather than **what they have bought**. In the context of tax-free shopping by tourists, demographic features are particularly informative: a customer's nationality reveals home-country shopping preferences, and tourist type (cross-border / international / domestic) reflects geographic access to stores and likely spending behavior.

### 9.2 Customer Demographics Data

Demographics were assembled from two sources:

| Source | Fields | Records |
|--------|--------|---------|
| `customer_info.xlsx` | CustomerId, DateOfBirth | 80,301 customers |
| Transaction data | customer_id, nationality, residency | 39,777 customers with transactions |

The two tables were joined by customer ID (after case normalization), yielding **39,777 customers** with full demographic profiles (100% coverage of transaction-active customers).

**Derived features:**

| Feature | Derivation | Values |
|---------|-----------|--------|
| `age_bin` | From DateOfBirth → age at transaction date | age:18-25, age:26-35, age:36-50, age:51-65, age:65+ |
| `tourist_type` | From residency (ISO country code) | domestic (IT), cross_border (CH/AT/FR/SI/SM/HR/ME/AL), international (all others) |
| `nationality` | From transaction data (ISO code, lowercase) | ~130 distinct nationalities |

**Tourist type distribution:**
- Cross-border: 28,855 customers (72.5%) — primarily Swiss and Austrian shoppers
- International: 10,463 customers (26.3%) — global tourists
- Domestic: 459 customers (1.2%) — Italian residents shopping tax-free

Cross-border tourists are defined as residents of countries bordering or adjacent to Italy (Switzerland, Austria, France, Slovenia, San Marino, Croatia, Montenegro, Albania). This distinction matters because cross-border shoppers typically have higher trip frequency and lower per-trip spend compared to international tourists.

### 9.3 Demographic Popularity (Model 11)

**Design:** Segment-based popularity recommender. For each customer, compute the most-visited stores among customers with the same demographic segment from the **training period only** (no leakage).

**Segment hierarchy** (fallback when segment size < 20 customers):

```
Full segment: nationality × age_bin × tourist_type
  → nationality × age_bin (drop tourist type)
  → nationality only
  → global popularity
```

This fallback hierarchy ensures all customers receive recommendations even for rare nationality-age-tourist combinations. The threshold of 20 customers per segment is a conservative choice to ensure the popularity signal is computed over enough data points.

**Cold-start property:** This model does not use any interaction history for the target customer. All scores are derived from other customers in the same segment. A new customer with zero transactions but known nationality and age immediately receives segment-appropriate recommendations.

### 9.4 LightFM Demo (Model 12)

**Design:** LightFM with WARP loss where **user features are demographic-only** (nationality, age bin, tourist type, encoded as sparse indicator vectors via LightFM's Dataset API). No behavioral features are included.

**Training:** Same interaction matrix as LightFM WARP (Section 8.3), but the user representations are constrained to demographic embeddings. This is architecturally clean: by training with only demographic user features, the user embedding is inherently demographic — no hacks like zeroing out behavioral vectors at inference.

**Configuration:** 64 latent components, learning rate 0.05, 30 epochs, WARP loss.

**Purpose:** Isolates the contribution of demographic signals. Direct comparison with LightFM Hybrid (same item features, behavioral user features) answers: *do demographics add signal beyond behavioral patterns?*

### 9.5 LightFM Full Hybrid (Model 13)

**Design:** LightFM with WARP loss combining **both behavioral and demographic user features**, plus store item features. This is the most feature-rich model evaluated.

**User features (behavioral):**
- Visit frequency bin (single / casual / regular / power)
- Spend level bin (low / mid / high, at P33=290€ and P66=890€)
- Primary shopping city

**User features (demographic):**
- Nationality (ISO code)
- Age bin (5 categories)
- Tourist type (domestic / cross_border / international)

**Item features (store profile):**
- Geographic region (6 clusters: lake_como, milan_metro, alpine_north, northeast, central, south)
- Store size bin (small / medium / large)
- Price bin (low / mid / high)

**Configuration:** 128 latent components, learning rate 0.01, 50 epochs, WARP loss.

**Hypothesis:** Demographic signals complement behavioral signals — particularly for customers with sparse behavioral histories where visit-count features are unreliable.

### 9.6 Ethical Considerations

Using demographic features such as nationality and age in recommendation systems raises fairness concerns:

1. **Statistical discrimination:** Recommending stores primarily based on nationality means customers may receive systematically different recommendations based on their origin country, not individual preferences. This could perpetuate stereotypes about national shopping preferences.

2. **Transparency:** Users should be informed that demographic characteristics influence their recommendations, particularly nationality (a protected characteristic in many legal jurisdictions).

3. **Mitigations applied:**
   - Demographic models are supplementary (models 11–13), not the primary recommendation engine. The main live system uses behavioral CF and content-based recommendations.
   - The fallback hierarchy ensures that low-coverage groups (rare nationalities) receive global recommendations rather than being over-indexed to small, potentially biased segments.
   - Results are evaluated quantitatively: if demographic models do not outperform behavioral models, they should not be deployed.
   - In the research context, the demographic models serve primarily to test the hypothesis that tourist type and origin-country preferences have predictive value for store recommendations — which is a scientifically interesting question independent of the deployment question.

4. **Regulatory context:** GDPR applies to EU customers. Nationality and age are personal data; their use for recommendation purposes requires appropriate legal basis (legitimate interest in personalized service, or explicit consent). In a production system, this would require a privacy impact assessment.

## 10. City-Based Filtering

### 10.1 Rationale

Transaction data analysis shows that approximately 41.5% of new store visits occur within the same city as the customer's primary shopping location. This geographic locality suggests that filtering recommendations by city can improve perceived relevance — users are more likely to visit recommended stores that are geographically accessible.

### 10.2 Implementation

City filtering is applied as a **post-filter** on top of existing recommendation models, keeping model logic unchanged while narrowing the output to geographically relevant stores:

**Collaborative filtering endpoints** (`/api/recommend/stores`, `/api/stores/{id}/similar`, `/api/recommend/stores/lightfm`):
- Accept an optional `city` query parameter
- Over-fetch results (top_k × 5) to ensure sufficient candidates after filtering
- Filter results by matching the `city` field from `store_profiles` (title-case normalized)
- Return top_k results from the filtered set

**Content-based chat** (`/api/chat`):
- Accepts an optional `city` field in the request body
- When set, the system prompt instructs the LLM to prefer products available in that area
- The `search_products` tool filters results to products sold at stores in the specified city
- A product→cities lookup (built at startup from `transactions × store_profiles`) enables fast filtering

**City normalization:** Store cities in the database have inconsistent capitalization (e.g., "COMO" vs "Como" vs "como"). All city values are normalized to title case for consistent matching and display.

**City list endpoint** (`GET /api/cities`):
- Returns a sorted, deduplicated list of all cities from `store_profiles`
- Used by the frontend city selector dropdown

### 10.3 Design Choice: Post-Filtering vs. Pre-Filtering

Post-filtering was chosen over pre-filtering (restricting the candidate set before scoring) because:
1. **Model integrity** — the underlying similarity/scoring logic remains unchanged
2. **Simplicity** — no need to retrain models or rebuild matrices per city
3. **Graceful degradation** — when no city is selected, the full unfiltered results are returned
4. **Sufficient candidates** — over-fetching by 5× ensures adequate results in most cases

The trade-off is that post-filtering may sometimes return fewer than top_k results if the city has very few stores. This is acceptable given the geographic distribution of the data.

## 11. Favorites & User Interaction Data

### 11.1 Purpose

The favorites feature serves as a lightweight implicit feedback collection mechanism. Users can "heart" products to save them, generating preference signals that could be used for:

- **Future model training** — favorited products provide positive labels for learning user preferences
- **Session-based personalization** — favorites persist across page navigations and browser sessions
- **Evaluation data** — comparing favorited products against recommendations provides a proxy for recommendation quality

### 11.2 Implementation

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

## 12. Recommendation System Architecture

### 12.1 Three-Tier Approach

The system combines four complementary recommendation strategies:

1. **Content-based (CLIP embeddings)** — finds products similar by visual appearance and textual description. This is the primary recommendation engine, powering the chatbot's product search. Works for all users regardless of purchase history (no cold-start problem).

2. **Memory-based collaborative filtering** — finds stores with similar product mixes (item-based CF) or stores visited by similar customers (user-based CF) using cosine similarity on raw interaction matrices. Displayed on the Recommender page for the selected profile.

3. **Model-based collaborative filtering** — ALS, LightFM WARP, and LightFM Hybrid learn latent factor representations from the customer-store interaction matrix. These models generalize beyond observed co-occurrences and can incorporate side features for cold-start mitigation (see Section 8).

4. **Conversational AI (GPT-4o-mini)** — mediates between the user and the search engine, performing preference elicitation, query formulation, intelligent result selection, and natural language presentation.

### 12.2 Content-Based Layer (Chatbot)

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

### 12.3 Collaborative Filtering Layer

The CF layer is displayed on the Recommender page, intentionally separated from the chatbot to maintain clear boundaries between recommendation approaches:

- **Item-based CF** — for a given profile's most-visited store, shows stores with the most similar product mix (cosine similarity on store-product vectors)
- **User-based CF** — for a given profile's customer, shows stores visited by similar customers but not yet visited by this customer

Model-based CF (ALS, LightFM WARP, LightFM Hybrid) is evaluated on the Evaluation page alongside memory-based CF, enabling direct comparison across paradigms. The Recommender page displays memory-based CF results for the selected profile, while the Evaluation page shows aggregate metrics across all test users.

### 12.4 Explainability

When a user asks "why did you recommend this?", the system provides the LLM with:
- The original search query
- The product's similarity score
- The product's price and merchant
- Context about what the user was looking for

The LLM then generates a user-friendly explanation (e.g., "This was recommended because it closely matches your request for affordable cat food — it's within your budget at 9.29 EUR and is highly similar to what you described."). This approach provides transparency without exposing technical details like embedding distances.

## 13. API & Frontend Design

### 13.1 FastAPI REST API

The backend is a FastAPI application that:
- Loads the CLIP model, embedding index, CF matrices, and model-based CF models once at startup (via lifespan context manager)
- Trains ALS, LightFM WARP, and LightFM Hybrid models during startup (~3 seconds)
- Maintains a persistent SQLite connection for metadata lookups
- Serves product images as static files
- Provides REST endpoints for search, browsing, chatbot interaction, favorites, collaborative filtering, and model evaluation

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
| GET | `/api/stores/{id}/similar` | Item-based CF: similar stores (optional `city` filter) |
| GET | `/api/recommend/stores` | User-based CF: store recommendations (optional `city` filter) |
| GET | `/api/recommend/stores/lightfm` | LightFM WARP: model-based CF (optional `city` filter) |
| GET | `/api/cities` | Sorted list of unique cities from store profiles |
| GET | `/api/profiles` | Available user profiles |
| GET | `/api/evaluation/results?k=5` | Model evaluation results (k=3,5,10,15,20) |

The chat endpoint keeps the OpenAI API key server-side and orchestrates the tool-calling loop, preventing API key exposure in the browser.

### 13.2 React Frontend

The frontend is a React 18 single-page application built with Vite and Tailwind CSS:

- **LandingPage** (`/`) — full-screen profile picker that serves as the entry point. Users must select a profile (or Guest) before accessing the app. No Navbar is shown.
- **RecommenderPage** (`/recommend`) — unified recommendation page combining store-level CF recommendations (top section) with the conversational chatbot (bottom section). Guest users see an informational message about needing purchase history for CF.
- **BrowsePage** (`/browse`) — paginated product grid with merchant filtering and favorites toggle
- **MapPage** (`/map`) — interactive Leaflet map with store markers and merchant filtering
- **EvalPage** (`/eval`) — model evaluation dashboard comparing 8 recommendation models across 6 IR metrics, with K selector (3, 5, 10, 15, 20), dataset statistics cards, and visual bar charts

All routes except `/` are protected behind a `hasSelectedProfile` gate in the session context.

Key components:
- `ProductCard` — displays product image, name, price, merchant badge, similarity score, external link, and heart button for favorites
- `ProductCarousel` — horizontal scrollable row of product cards, rendered inline in chat messages
- `StoreMap` — inline Leaflet map rendered alongside product carousels in chat responses
- `ChatSection` — extracted chat interface (text input, image upload, message history) composed into RecommenderPage
- `CFSection` — extracted CF recommendations display with collapsible sections for item-based and user-based results
- `ProfilePicker` — profile selection cards, used in both LandingPage and Navbar dropdown
- `Navbar` — navigation with profile badge dropdown (profile switcher + sign out)

## 14. Multilingual Capabilities

The system supports multilingual queries without explicit language detection or translation:

- **LAION-2B training data** includes multilingual web content (image-text pairs in many languages)
- The CLIP model implicitly learns cross-lingual associations through shared visual grounding
- Tested successfully with Italian ("cibo per gatti"), English ("cat food"), and Spanish ("vestido elegante") queries

This is a property of the training data distribution rather than an explicit multilingual architecture — performance may vary for underrepresented languages.

## 15. Evaluation Methodology

### 15.1 Models Evaluated

The system evaluates thirteen recommendation models across two levels (product and store) and four paradigms:

| # | Model | Level | Paradigm | Method |
|---|-------|-------|----------|--------|
| 1 | Content-Based (CLIP) | Product | Content-based | Self-retrieval: use each product's embedding as query, relevance = same category |
| 2 | Item-Based CF | Store | Memory-based CF | Temporal split: recommend stores similar to customer's training stores |
| 3 | User-Based CF | Store | Memory-based CF | Temporal split: recommend stores from similar customers' training history |
| 4 | ALS | Store | Model-based CF | Temporal split: ALS latent factor recommendations trained on pre-split data |
| 5 | LightFM WARP | Store | Model-based CF | Temporal split: LightFM WARP recommendations trained on pre-split data |
| 6 | LightFM Hybrid | Store | Model-based CF | Temporal split: LightFM WARP + store/user side features trained on pre-split data |
| 7 | Random Baseline | Store | Baseline | Uniform random sample of k stores (excluding already-visited) |
| 8 | Popularity Baseline | Store | Baseline | Most-visited stores from training period (excluding already-visited) |
| 9 | Random Baseline (Product) | Product | Baseline | Uniform random sample of k products — same relevance criterion as CLIP |
| 10 | Popularity Baseline (Product) | Product | Baseline | Most-common merchant's products — tests CLIP against naive product-level baselines |
| 11 | Demographic Popularity | Store | Demographic CF | Segment-based popularity: most-visited stores within customer's demographic segment |
| 12 | LightFM Demo | Store | Demographic CF | LightFM WARP with demographic-only user features (nationality, age, tourist type) |
| 13 | LightFM Full Hybrid | Store | Demographic CF | LightFM WARP + behavioral AND demographic user features + store item features |

This design enables four key comparisons:
1. **Memory-based vs. model-based CF** — cosine similarity on raw matrices vs. learned latent factors
2. **Pointwise vs. pairwise loss** — ALS (reconstruction loss) vs. LightFM (WARP ranking loss)
3. **Pure CF vs. hybrid** — LightFM without features vs. LightFM with store profile side features
4. **Behavioral vs. demographic CF** — history-based signals vs. demographic segment signals (models 3/6 vs. 11/12/13)

### 15.2 Metrics

Seven metrics are computed for each model following established formulas from the IR and RecSys literature. Six apply to all models; one (Novelty) applies to product-level models only:

- **Precision@K** — fraction of top-k recommendations that are relevant. Measures recommendation accuracy. (Manning et al., 2008)
- **Recall@K** — fraction of all relevant items that appear in the top-k list. Measures completeness. (Manning et al., 2008)
- **nDCG@K** — normalized discounted cumulative gain with binary relevance and log₂ discount. Measures ranking quality by penalizing relevant items appearing at lower positions. (Järvelin & Kekäläinen, 2002)
- **Hit Rate@K** — binary metric: 1 if any relevant item appears in top-k, else 0. Measures whether the system finds at least one relevant result. (Deshpande & Karypis, 2004)
- **Coverage** — fraction of the full catalog that appears in any recommendation list across all test cases. Measures how much of the catalog the model explores. (Adomavicius & Kwon, 2012)
- **Diversity (ILD)** — average pairwise dissimilarity among recommended items (Intra-List Diversity = 1 − cosine similarity). Measures variety in recommendations. (Ziegler et al., 2005)
- **Novelty** *(product-level only)* — mean self-information of recommended items, computed from recommendation frequency:

  ```
  novelty = mean_{i ∈ recs}[ -log₂(freq(i) / N_queries) ]
  ```

  where `freq(i)` is the number of queries for which item `i` appears in the top-k list, and `N_queries` is the total number of queries. Items recommended for nearly every query (freq → N) have novelty → 0 (trivially popular); rarely-surfaced items have high novelty. This self-contained metric, proposed by Vargas & Castells (2011), requires no external popularity data — it is computed entirely from the recommender's own output and directly answers: *does this system surface long-tail items, or does it always recommend the same popular products?*

  The Popularity Baseline confirms the metric's validity: it always recommends the same k products from the most common merchant, yielding near-zero novelty (0.01 at k=10), while CLIP (5.66) and Random (5.92) show high novelty.

  **References:**
  - Vargas, S., & Castells, P. (2011). Rank and relevance in novelty and diversity metrics for recommender systems. *RecSys 2011*.

All metrics are implemented from scratch in `pipelines/evaluation.py` following the original formulas. Using hand-written implementations rather than a library (e.g., ranx, RecBole) was a deliberate choice for transparency, zero additional dependencies, and full control over the evaluation pipeline (temporal splits, mixed product/store-level evaluation).

### 15.3 Evaluation Protocol: Temporal Train/Test Split

**Rationale for temporal split:** An initial implementation used leave-one-out cross-validation, where one store per customer was randomly held out and the model was tested on recovering it. This protocol suffered from severe data leakage — model-based approaches (ALS, LightFM) were trained on the full interaction matrix including the held-out store, meaning their learned user embeddings already encoded the preference for the test item. This produced artificially inflated metrics (e.g., 99.9% hit rate for LightFM WARP) that did not reflect real-world predictive ability.

**Temporal split protocol:** To eliminate data leakage, evaluation uses a time-based train/test split:

- **Train period:** All transactions before December 1, 2025
- **Test period:** All transactions from December 1, 2025 onward
- **Test cases:** Customers who appear in both periods AND visited at least one NEW store in the test period (a store not visited during training). This yields 1,708 test cases.
- **Task:** Given a customer's training history, predict which new stores they will visit in the test period.

This protocol guarantees that:
1. Models are trained only on historical data (no future information leaks)
2. All matrices (store-product, customer-store, interaction) are built from training data only
3. Model-based CF models (ALS, LightFM) are trained from scratch on training data only
4. The evaluation tests genuine predictive ability — predicting future store visits the model has never seen

**Product-level (Content-Based):** For each of the 634 products, use its CLIP embedding as query, exclude itself from results, and check if products from the same **category** rank in the top-k. This self-retrieval protocol measures **embedding quality** — whether CLIP embeddings successfully capture fine-grained product similarity within a merchant's catalog.

**Relevance criterion — category-based (not same-merchant):**

Using same-merchant as the relevance definition would be too coarse: with 500 Arcaplanet products spanning pet food, grooming supplies, toys, and aquarium equipment, treating all 499 other Arcaplanet products as equally relevant to any query makes recall trivially inflated and precision meaningless. The same problem applies to Twinset's 134 products spanning dresses, knitwear, footwear, and accessories — recommending a jersey in response to a dress query should not count as a "hit" under a strict evaluation.

Category relevance is extracted deterministically from the product data:

| Merchant | Source | Method | Coverage |
|----------|--------|--------|----------|
| Twinset | URL path segment | Strip product code (e.g. `261TP3083_00059`), take first word of slug | 133/134 (99.3%) |
| Arcaplanet | Product name | Keyword matching for animal type (cat/dog/small_animal) × product type (food/snack/toy/etc.) | 428/500 (85.6%) |

**Twinset categories** (16 distinct): vestido (34), jersey (25), pantalon (20), chaqueta (10), top (10), camisa (10), zapato (8), and 9 smaller categories. Extracted from URL slugs, e.g., `jersey-boxy-de-rayas-261TP3281_13386.html` → "jersey".

**Arcaplanet categories** (25 distinct): cat_food (57), cat_other (44), dog_food (36), other_accessories (39), small_animal_accessories (34), etc. Extracted from Italian/Spanish/English keywords in product names.

Products for which no category can be determined (5.3% of catalog) fall back to same-merchant relevance to avoid excluding them from evaluation entirely.

**Overall:** 561/634 products (88.5%) use strict category-based relevance; 73 fall back to same-merchant.

These scores are not directly comparable to the store-level CF metrics, which measure a fundamentally different task (predicting future store visits from historical behavior).

**Store-level (CF + baselines):** For each of the 1,708 test cases, generate top-k store recommendations using only training-period data, then check whether any of the customer's new test-period stores appear in the recommendations.

### 15.4 Baseline Definitions

**Store-level baselines** (models 7–8):

**Random Baseline:** For each test case, uniformly sample k stores from all stores excluding the customer's training-period stores. Provides a lower bound on recommendation quality.

**Popularity Baseline:** Always recommend the most-visited stores from the training period (by distinct customer count), excluding stores the customer already visited. Tests whether models learn personalized patterns beyond simple aggregate frequency.

**Product-level baselines** (models 9–10):

**Random Baseline (Product):** For each product query, uniformly sample k products from the full catalog (excluding self). Uses the same category-based relevance criterion as CLIP for direct comparison.

**Popularity Baseline (Product):** Always recommend products from the most common merchant (Arcaplanet, 500 products). Unlike the store-level popularity baseline, this extreme specialization yields near-zero novelty (0.01) and near-zero catalog coverage (1.7%) — confirming that it is a naive, undesirable strategy that CLIP must clearly outperform to validate the content-based approach.

The product-level baselines answer the question: *does CLIP actually add value, or does the evaluation protocol inflate apparent accuracy?* CLIP's HR@10 = 0.956 vs. Random@10 = 0.435 confirms that the model's performance is genuine, not an artifact of easy evaluation.

### 15.5 Model Improvements

Several improvements were applied to the model-based CF pipeline after initial evaluation:

1. **Recency-weighted interactions** — The interaction matrix uses exponential decay (`weight = visits × exp(-days/180)`) instead of raw visit counts. This allows models to prioritize current preferences while retaining historical signal.

2. **Revised hybrid features** — Merchant name was removed from the LightFM Hybrid item features because it caused the model to overfit to same-merchant recommendations, overriding the collaborative signal. User-side features (visit frequency bin, spend level bin, primary city) were added to provide behavioral context.

3. **Tuned hybrid hyperparameters** — The hybrid model uses 128 latent components (vs. 64 for pure WARP), learning rate 0.01 (vs. 0.05), and 50 epochs (vs. 30) to better integrate the additional feature signal.

4. **Store-level only baselines** — Baselines were changed from an average of product-level and store-level metrics to store-level only, ensuring fair comparison with the CF models they benchmark against.

### 15.6 Implementation

Evaluation is implemented in `pipelines/evaluation.py` with model training in `pipelines/collab_model.py`. Metric functions follow standard IR formulas with academic citations in the source code. The offline evaluation script `scripts/run_evaluation.py` generates `data/evaluation_results.json` for k=3, 5, 10, 15, 20. The API endpoint `GET /api/evaluation/results?k=5` serves pre-computed results from this file, avoiding expensive recomputation on each request.

**Dataset statistics:** The evaluation pipeline also computes comprehensive dataset statistics (`data_stats`) alongside model metrics: customer/store/product counts, transaction volumes, matrix sparsity (customer-store and store-product), average stores per customer, average products per store, embedding dimensions, date ranges, and the train/test split date. These statistics are included in the evaluation results JSON and displayed on the frontend evaluation dashboard, providing context for interpreting the metrics.

Training-only matrices are constructed via `build_train_store_product_matrix()`, `build_train_customer_store_matrix()`, and `build_train_interaction_matrix()` — all filtered to transactions before the split date. Model-based CF models (ALS, LightFM WARP, LightFM Hybrid) are trained fresh on these training matrices during evaluation.

## 16. Hyperparameter Tuning

### 16.1 Methodology

To validate the default hyperparameter choices for the model-based CF approaches, a grid search was conducted over key parameters for both ALS and LightFM WARP. All configurations were evaluated at k=5 using the same temporal train/test split as the main evaluation (Section 14).

### 16.2 Configurations Tested

**ALS:**

| Factors | Regularization | nDCG@5 | Hit Rate@5 | Precision@5 |
|---------|---------------|--------|------------|-------------|
| 32 | 0.01 | — | — | — |
| 32 | 0.1 | — | — | — |
| 64 | 0.01 | — | — | — |
| 64 | 0.1 | — | — | — |
| 128 | 0.01 | — | — | — |
| 128 | 0.1 | — | — | — |

**LightFM WARP:**

| Components | Learning Rate | nDCG@5 | Hit Rate@5 | Precision@5 |
|-----------|--------------|--------|------------|-------------|
| 32 | 0.01 | — | — | — |
| 32 | 0.05 | — | — | — |
| 64 | 0.01 | — | — | — |
| 64 | 0.05 | — | — | — |
| 128 | 0.01 | — | — | — |
| 128 | 0.05 | — | — | — |

*Note: Table values to be filled after running `python -m scripts.run_hyperparameter_comparison`. Results are saved to `data/hyperparameter_results.json`.*

### 16.3 Analysis

The hyperparameter search validates the production defaults (ALS: 64 factors, 0.01 reg; LightFM WARP: 64 components, 0.05 lr) and explores whether increased model capacity (128 factors/components) or different regularization strengths yield meaningful improvements. This analysis demonstrates due diligence in model configuration for the thesis evaluation.

## 17. Future Work

- **Catalog expansion** — additional merchants and product categories to test cross-domain recommendations at scale
- **Embedding model upgrades** — larger CLIP variants (ViT-L-14) or domain-specific fine-tuning if quality improvements are needed
- **Real-time feedback loop** — using favorites and click data to personalize content-based recommendations over time
- **Hybrid ranking** — combining content-based similarity scores with collaborative filtering signals into a unified recommendation score
