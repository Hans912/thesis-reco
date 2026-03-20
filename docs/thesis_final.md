# Designing a Conversational Travel Assistant for Personalized Store and Product Recommendations within a Tax-Free Ecosystem

**Student:** Hans Helmrich
**Degree:** Double Degree in Business Administration & Business Data Analytics
**Supervisor:** Prof. Santiago Gil Begue
**Institution:** IE University, School of Science & Technology

---

## Abstract

Tax-free shopping ecosystems present a distinct challenge for personalization: tourists visiting the Como and Milan area of northern Italy typically interact with the system only once or twice, producing interaction histories too sparse for conventional collaborative filtering. Simultaneously, the ecosystem spans two partner merchants with visually and semantically distinct product catalogs — a fashion retailer and a pet supply retailer — which must be searchable through a single unified interface. This thesis designs and evaluates a conversational travel assistant that addresses these constraints through three integrated components: OpenCLIP multimodal embeddings for product discovery, store-level collaborative filtering for itinerary recommendations, and a GPT-4o-mini tool-calling layer for natural language interaction.

The product-level evaluation covers 634 products across both merchants and uses category-based relevance to measure embedding quality. OpenCLIP ViT-B-32 achieves HR@10 = 0.956 and nDCG@10 = 0.693, well above both a random baseline (HR@10 = 0.435) and a popularity baseline (HR@10 = 0.456). The store-level evaluation employs a temporal split over 1,708 test customers, measuring how well 13 models predict store visits made after December 1, 2025. LightFM with WARP loss achieves HR@10 = 0.248 with default hyperparameters and HR@10 = 0.318 after Bayesian hyperparameter optimization (Optuna, 40 trials), a sixteen-fold improvement over the random baseline (0.019) and a 105% improvement over global popularity (0.155). Demographic features — nationality, tourist type, and age — provide cold-start capability competitive with behavioral collaborative filtering, though not superior to model-based approaches trained on interaction data.

The primary contributions are: a working end-to-end web application integrating all three components; a systematic 13-model evaluation with a leak-free temporal protocol; a multimodal embedding ablation study and natural language query evaluation confirming the validity of the fused image-text design; a structured conversational evaluation across 25 scenarios; and an empirical analysis of recommendation quality under extreme sparsity (99.81% of the customer–store matrix is unobserved). The results demonstrate that shared image–text embeddings transfer effectively to specialized multilingual retail catalogs, and that LightFM WARP is the most appropriate store recommendation approach at this level of data scarcity.

---

## Chapter 1: Introduction

### 1.1 Context

Northern Italy's Lake Como and Milan region attracts millions of international visitors each year. For tourists whose country of residence is outside the European Union, purchases at eligible stores generate a VAT refund through the tax-free shopping mechanism. This mechanism is administered by intermediary platforms that process invoices, verify eligibility, and distribute refunds. The platform studied in this thesis operates across 776 stores in the northern Italian region, processing transactions for roughly 40,000 customers annually.

From a traveler's perspective, the tax-free benefit creates an incentive to shop within the ecosystem — but only at stores that participate. A tourist unfamiliar with the region faces a practical challenge: short visit windows, an unfamiliar retail landscape, language barriers, and no systematic way to discover which stores carry products relevant to their interests. General-purpose travel tools such as mapping applications or review platforms are not ecosystem-aware; they do not know which stores are tax-free eligible, which stores are currently active, or which stores match a given traveler's preferences based on their prior behavior.

Despite this clear opportunity, no personalized digital assistant exists within this ecosystem. Travelers currently rely on in-store staff, word of mouth, or generic search engines — none of which incorporate the ecosystem's unique structure of verified merchants, refund eligibility, and transaction history.

This thesis responds to that gap by designing a conversational travel assistant that operates within the tax-free ecosystem, combining product-level search with store-level recommendations in a natural language interface.

### 1.2 Problem Statement

Building a personalized recommendation system for this context requires navigating three technical challenges that compound one another.

The first challenge is cold-start. Most tourists visit stores only once or twice during a trip. Of the 39,777 customers in the dataset, the average number of distinct store visits is 1.42, meaning the vast majority of users arrive with no meaningful interaction history, and even the most active users have visited only a small fraction of the 776 stores in the network. Conventional collaborative filtering assumes that users have expressed preferences across multiple items; when preferences are reduced to a single data point, most CF approaches degrade toward random or popularity-based recommendations.

The second challenge is sparsity. The customer–store interaction matrix is 99.81% empty, and the store–product matrix is 99.83% empty. These sparsity levels are extreme even by the standards of retail recommendation systems, which routinely deal with sparse data but rarely at this density. Memory-based collaborative filtering, which computes pairwise similarity directly from the raw matrix, is particularly vulnerable: cosine similarity between two near-empty vectors is numerically unreliable and often dominated by noise.

The third challenge is cross-merchant product discovery. Two partner merchants contribute product catalogs: a fashion retailer with 134 products and a pet supply retailer with 500 products. These catalogs differ in every relevant dimension — visual appearance, textual description language (Italian versus Spanish), price range, and product semantics. A traveler searching for "something for my dog" should retrieve pet supply products; a traveler searching for "a dress for a summer evening" should retrieve fashion items. A unified search interface must handle both without requiring the user to specify which merchant to search. Classical text retrieval methods such as TF-IDF struggle with this because product names and descriptions use different vocabularies, languages, and levels of detail across merchants.

### 1.3 Research Questions

Four research questions structure the empirical contributions of this thesis:

**RQ1:** Can multimodal CLIP embeddings enable effective cross-merchant product recommendations in a sparse, multilingual retail catalog?

**RQ2:** Can store-level collaborative filtering produce meaningful personalized store recommendations despite extreme sparsity?

**RQ3:** Do demographic features — nationality, tourist type, and age — improve store recommendation quality over purely behavioral approaches?

**RQ4:** Can a tool-calling large language model effectively mediate between user preferences and a CLIP search index for conversational product discovery?

### 1.4 Contributions

This thesis makes four concrete contributions.

First, a working conversational travel assistant. The system integrates OpenCLIP product search, 13 store recommendation models, and a GPT-4o-mini conversational interface into a deployed web application. The backend is a FastAPI server; the frontend is a React application. The full source code implements the pipeline from raw transaction data through to served recommendations.

Second, a systematic 13-model evaluation. Product-level and store-level evaluations are conducted with seven metrics: Hit Rate, nDCG, Precision, Recall, Coverage, Intra-List Diversity, and Novelty. The store-level evaluation uses a rigorously leak-free temporal split, correcting a data leakage flaw that leave-one-out cross-validation introduced in an earlier iteration of the system.

Third, an empirical analysis of CF under extreme sparsity. The evaluation quantifies how model-based approaches (ALS, LightFM) compare to memory-based CF and demographic models under a 99.81% sparse interaction matrix, providing specific guidance on which model families are viable when average user history is a single store visit.

Fourth, a demographic augmentation study. Three demographic models — Demographic Popularity, LightFM Demo (demographic features only), and LightFM Full Hybrid (behavioral plus demographic) — are evaluated against purely behavioral alternatives, isolating the marginal contribution of nationality and tourist-type signals.

### 1.5 Thesis Structure

Chapter 2 reviews the relevant literature, covering recommendation systems, conversational recommendation, context-aware and location-based approaches, multimodal embeddings, and the tax-free shopping domain. Chapter 3 describes the data sources, their structure, and the challenges encountered in combining them. Chapter 4 presents the methodology: the CLIP embedding pipeline, five collaborative filtering approaches, three demographic models, and the conversational AI layer. Chapter 5 describes the evaluation design in detail, including the temporal split protocol and the category-based relevance judgment method. Chapter 6 presents and discusses the numerical results. Chapter 7 discusses the implications, limitations, and directions for future work. Chapter 8 concludes.

---

## Chapter 2: Literature Review

### 2.0 Introduction

International travelers navigating an unfamiliar retail environment face a particular kind of information asymmetry: they may know what they want but have no reliable way to identify which nearby stores carry it, which stores are worth their limited time, or how to communicate their preferences across a language barrier. Tax-free shopping ecosystems layer an additional constraint on this: only participating merchants qualify for the VAT refund, so discovery must happen within a constrained but structured network. These conditions — short visit windows, cold-start users, multilingual content, and a defined merchant set — point toward a specific configuration of recommendation technologies. This chapter reviews the academic foundations for that configuration, covering recommendation systems broadly, conversational recommendation, context-aware approaches, multimodal embedding methods, and the tax-free shopping domain.

### 2.1 Recommendation Systems as Decision-Support Tools

Recommendation systems reduce information overload by generating ranked lists of items predicted to be relevant to a specific user at a specific moment. Schafer et al. (2001) identified recommendation as a core e-commerce function, demonstrating that automated filtering creates measurable commercial value by surfacing relevant items that users would otherwise not discover. Stalidis et al. (2023) provide a contemporary review of recommendation systems for retail and e-commerce, confirming that personalized recommendation remains central to digital commerce strategy across sectors including fashion, electronics, and food.

In the travel context, the decision-support framing is particularly apt. A tourist deciding which stores to visit is making a sequential discovery problem under time pressure and partial information. Recommendation systems can structure this problem by presenting a ranked shortlist — reducing the effective search space from hundreds of stores to a manageable set of candidates.

### 2.2 Core Approaches: Content-Based, Collaborative, and Hybrid Filtering

Adomavicius and Tuzhilin (2005) provide the canonical taxonomy of recommendation approaches. Content-based filtering generates recommendations by matching item attributes to user preference profiles. Items are described by feature vectors; users are represented by the aggregate features of items they have previously engaged with; recommendations are items whose features best match the user profile. This approach is effective when item descriptions are rich and discriminative, and when user preferences can be reliably inferred from content features.

Collaborative filtering (CF) generates recommendations by identifying users with similar preference histories (user-based CF) or items that co-occur in similar users' histories (item-based CF). The core insight is that behavioral similarity is informative even when item content is not directly comparable. Adomavicius and Tuzhilin (2005) distinguish between memory-based CF, which computes recommendations directly from stored interaction data using similarity metrics, and model-based CF, which learns latent representations of users and items that generalize beyond observed interactions. Herlocker et al. (2004) provide a thorough treatment of evaluation methodologies for CF systems, identifying the conditions under which different protocols are appropriate.

Burke (2002) introduced the hybrid recommender framework, showing that combining content-based and collaborative filtering typically outperforms either approach alone. The central argument is that the two approaches fail in complementary ways: content-based filtering struggles when item descriptions are poor or when users want serendipitous discovery beyond their stated preferences; collaborative filtering struggles when interaction data is sparse or when new items have no engagement history. Hybrid models, including those that merge content features with learned interaction embeddings, can address both failure modes simultaneously. Stalidis et al. (2023) confirm that hybrid architectures dominate the contemporary literature for complex retail environments.

Isinkaye et al. (2015) and Javed et al. (2021) survey the broader recommendation landscape, including context-aware and knowledge-based systems, and conclude that no single approach dominates across all domains. The appropriate architecture depends critically on data availability, catalog characteristics, and user behavior patterns.

### 2.3 The Cold-Start Problem

The cold-start problem refers to the failure mode of CF systems when a new user or item has insufficient interaction history to produce reliable similarity estimates. Son (2016) reviews approaches to new-user cold-start, categorizing solutions into interview-based methods (explicit preference elicitation), social network-based methods, and hybrid content-demographic methods. Lika et al. (2014) study cold-start more broadly, showing that the transition from cold to warm behavior typically requires between 5 and 20 interactions, far more than most new users in a tourist shopping context will accumulate.

The cold-start problem is especially acute in this thesis's setting, where 79.5% of customers (after expansion to two years of data) have a single transaction and the average is 1.42 store visits. At this level of data availability, pure CF provides almost no signal for individual users. The literature identifies three main mitigations: (1) demographic features, which allow segment-level predictions when individual history is absent; (2) content-based filtering, which requires no interaction history; and (3) conversational preference elicitation, which collects explicit preferences during the interaction itself.

### 2.4 Diversity, Novelty, and User Trust

Evaluation of recommendation systems in academic research has historically focused on accuracy metrics — hit rate, precision, recall, and normalized discounted cumulative gain. However, a growing body of research argues that accuracy alone is an incomplete criterion for recommendation quality, and that diversity, novelty, and transparency are essential dimensions of a well-functioning system.

Diversity in recommendation lists refers to the degree to which recommended items differ from one another. Ziegler et al. (2005) introduced topic diversification as a post-processing technique, showing that reducing intra-list similarity increases user satisfaction even when accuracy metrics decline slightly. Their work established the empirical basis for the accuracy-diversity tradeoff: lists that are maximally accurate — always recommending the single most-likely-to-be-relevant item — tend to be repetitive, offering the user little new information beyond confirming what they already know. Adomavicius and Kwon (2012) extended this framework to ranking-based diversification, showing that diversity can be improved through list construction strategies rather than only through post-hoc reranking.

Novelty is a related but distinct concept. A novel recommendation is one that the user is unlikely to have encountered through other channels — it represents genuine discovery rather than the confirmation of already-known options. Vargas and Castells (2011) formalize novelty as the self-information of a recommended item given its popularity in the recommendation distribution, operationalizing it as $-\log_2(p(i))$ where $p(i)$ is the proportion of queries for which item $i$ appears in the recommendation list. Under this formulation, items that appear in nearly every recommendation list approach novelty zero, while items surfaced only rarely carry high novelty scores. The popularity baseline in this thesis achieves near-zero novelty (0.012) by construction — it always recommends the same top stores — which validates the metric's discriminative power.

The relationship between accuracy and novelty is not always a tradeoff. A content-based model that retrieves genuinely similar items from across the full catalog may achieve both high accuracy and moderate-to-high novelty if the catalog is diverse. The CLIP model evaluated in this thesis achieves a novelty score of 5.662, compared to 5.918 for the random baseline and 0.012 for the popularity baseline — demonstrating that high accuracy does not require sacrificing novelty.

Beyond diversity and novelty, transparency and explainability are increasingly recognized as critical for user acceptance. Jannach et al. (2021) note that "explaining recommendations is considered a key feature to make decision-making easier or to increase user trust." When users understand why an item was recommended — because it is visually similar to a product they described, or because shoppers with similar profiles purchased it — they are better positioned to evaluate the recommendation's relevance and to provide informed feedback. In a tourist shopping context, where users may have little prior familiarity with the stores or products being recommended, transparency is particularly valuable. The conversational architecture developed in this thesis supports transparency through the explain_recommendation tool, which generates plain-language justifications for product recommendations.

### 2.5 Conversational Recommendation and Natural Language Interaction

#### 2.5.1 From Chatbots to Conversational Recommender Systems

The integration of natural language processing with recommendation systems has produced a distinct research area: conversational recommender systems (CRS). Jannach et al. (2021) provide a comprehensive survey, tracing the evolution from simple slot-filling chatbots — which ask users to fill in structured preference forms through constrained dialogue — to end-to-end neural systems capable of open-domain conversation about item properties. Gao et al. (2021) identify three functional capabilities that distinguish CRS from conventional recommenders: the ability to elicit fine-grained preferences through dialogue, the ability to revise recommendations based on user feedback, and the ability to explain recommendations in natural language.

Early CRS research focused on knowledge-based systems, which used structured ontologies of item properties to guide preference elicitation. These systems were effective but brittle: they required extensive manual knowledge engineering and could only handle queries that fit the predefined ontology. The availability of large language models has fundamentally changed the design space, enabling systems that handle free-form natural language input without requiring a fixed preference template.

#### 2.5.2 Interaction Patterns in CRS

Gao et al. (2021) and Jannach et al. (2021) identify three main interaction patterns in conversational recommender systems. In questioning-based CRS, the system proactively asks users about preferences — "Are you looking for something for yourself or as a gift?" — to refine the recommendation space. In critiquing-based CRS (Luo et al., 2020), the user evaluates an initial recommendation and provides directional feedback — "something cheaper" or "more formal" — and the system adjusts accordingly. In conversation-based CRS (Sun and Zhang, 2018), the recommendation emerges organically from a multi-turn dialogue that may begin with topic discovery and progressively narrow toward specific items.

The system developed in this thesis combines questioning and critiquing. The LLM asks clarifying questions when the user's initial query is ambiguous (questioning), and users can refine recommendations through follow-up messages that the LLM interprets as directional constraints (critiquing). Full open-domain conversation is supported through the underlying language model but is not the primary interaction mode.

#### 2.5.3 Evaluation of CRS

Evaluating conversational recommender systems is more complex than evaluating conventional recommenders. Jannach et al. (2021) argue that standard IR metrics are insufficient because they measure only the quality of the final recommendation list, ignoring the quality of the dialogue that produced it. Sun and Zhang (2018) propose metrics such as turn-level task success, dialogue efficiency (number of turns to reach a satisfactory recommendation), and user effort. In practice, these metrics require either user studies with real participants or carefully labeled dialogue datasets. This thesis does not conduct a formal CRS evaluation due to the absence of user study infrastructure; the conversational component is evaluated qualitatively rather than quantitatively. This is an acknowledged limitation discussed in Chapter 7.

### 2.6 Context-Aware and Location-Based Recommenders

#### 2.6.1 Why Context Matters

Adomavicius et al. (2011) define context-aware recommendation systems as systems that incorporate contextual information — time, location, social setting, device — into the recommendation process alongside user and item features. Their taxonomy of context integration approaches distinguishes pre-filtering (restricting the item set before applying a recommendation algorithm), post-filtering (applying the recommendation algorithm and then adjusting results based on context), and contextual modeling (incorporating context directly into the model's similarity or scoring function).

In travel and tourism, location is a first-order constraint. A store that is physically accessible within a tourist's itinerary is qualitatively different from a store that requires a separate trip. Recommending a store 50 kilometers from the user's current location is less useful than recommending a nearby alternative with slightly lower predicted relevance. The find_nearby_stores tool in the conversational assistant addresses this constraint by incorporating geographic proximity into store recommendations, implemented as a post-filtering step on top of the CF scores.

The broader tourism recommendation literature confirms the importance of location integration. Hwangbo et al. (2018) study fashion recommendation in e-commerce, finding that contextual filtering based on session behavior improves conversion rates. While this work focuses on online rather than physical retail, the underlying principle — that recommendation quality degrades when context is ignored — applies directly to the tourist shopping scenario.

### 2.7 Multimodal Embeddings for Product Representation

#### 2.7.1 CLIP and the Shared Embedding Space

Radford et al. (2021) introduced CLIP (Contrastive Language-Image Pre-Training), a model trained on 400 million image-text pairs scraped from the internet using a contrastive objective: the model learns to assign high similarity scores to matching image-text pairs and low scores to non-matching pairs. The result is a shared embedding space where images and text are directly comparable — a text query can retrieve images, and an image can retrieve text descriptions, without any task-specific fine-tuning.

For retail product search, the shared embedding space solves a specific problem: product catalogs contain both visual and textual information, and users may express preferences in either modality. A text query ("black leather jacket") should retrieve products that are visually black leather jackets, even if those products' text descriptions use different vocabulary or a different language. Conversely, an image uploaded by a user should retrieve textually described products that are visually similar. CLIP's contrastive pre-training makes this cross-modal retrieval possible without requiring paired training data from the specific retail domain.

#### 2.7.2 OpenCLIP and Large-Scale Pre-Training

Cherti et al. (2023) present OpenCLIP, an open reproduction of CLIP trained on LAION-2B, a dataset of 2 billion image-text pairs with broader coverage than the original 400 million pairs. The key finding is that scale matters: larger training datasets consistently improve downstream zero-shot performance, and the ViT-B-32 model trained on LAION-2B achieves strong zero-shot transfer to specialized domains including retail product images.

The LAION-2B dataset's multilingual component is relevant to this thesis. Product names and descriptions in the catalog are in Italian and Spanish; CLIP models trained primarily on English web data may not generalize well to multilingual content. The LAION-2B training corpus includes substantial non-English web content, which improves cross-lingual transfer. This makes OpenCLIP a better choice for a multilingual retail application than CLIP trained on the original English-dominated dataset.

#### 2.7.3 Why CLIP Outperforms Text-Only Approaches

Classical text-based product search methods such as TF-IDF or bag-of-words representations face several limitations in this setting. First, they cannot bridge the vocabulary gap between Italian, Spanish, and English product descriptions. A user querying "summer dress" in English will not retrieve products described as "vestido de verano" using TF-IDF unless explicit translation is performed. Second, text-only methods cannot use visual information: two products described as "pantalon negro" may look very different visually, and text similarity cannot distinguish them. Third, short product descriptions (frequently just a name and a brief descriptor) provide insufficient text for reliable similarity computation.

Word embeddings (Word2Vec, GloVe, FastText) address the vocabulary problem partially but still require language-specific training or multilingual models, and they do not incorporate visual information. Sentence transformers (SBERT) improve over word embeddings for semantic text similarity but similarly cannot leverage product images.

CLIP addresses all three limitations simultaneously: the contrastive objective aligns multilingual text and images in a single space, the visual encoder captures fine-grained visual product attributes, and the 512-dimensional shared embedding provides sufficient expressiveness for within-category similarity. The preliminary validation experiment described in Chapter 4 — querying with a text description and ranking real product images — confirms that this transfer works correctly for the specific product types in this catalog.

### 2.8 Tax-Free Shopping Ecosystem and Strategic Relevance

#### 2.8.1 Tax-Free Rules and Intermediary Market Structure

The European Union's VAT refund mechanism for non-EU residents creates a structured incentive for tax-free shopping. Under EU tax-free rules, travelers resident outside the EU may claim a refund of the VAT paid on goods exported from an EU country (European Commission, n.d.). The refund is processed by tax-free shopping intermediaries — companies that operate the invoice processing, customs validation, and payment infrastructure. Maffini and Ricci (2019) document the scale and structure of this market, noting that tax-free purchases represent a significant share of luxury and fashion retail revenue in Southern Europe, with the Como and Milan area being among the highest-value corridors in Italy.

The intermediary's role creates a specific loyalty dynamic. Travelers who use the platform's app or card for one transaction are incentivized to use it for subsequent transactions in the same trip, since the refund is aggregated across all qualifying purchases. However, the traveler has no particular reason to prefer one tax-free provider over another at the point of store entry — the choice is often determined by which provider is displayed in the store's POS terminal. A personalized digital assistant that actively helps the traveler discover and navigate the ecosystem creates a stickiness that the bare refund mechanism does not. If a traveler finds a new store through the platform's recommendation interface and that purchase qualifies for a refund, the platform captures a transaction it would otherwise not have processed.

#### 2.8.2 Competitive Dynamics and Differentiation Through User Experience

The tax-free intermediary market is mature and concentrated. Established incumbents hold large merchant networks built over decades, and their scale creates network effects: merchants prefer large-network providers because they reach more travelers, and travelers prefer large-network providers because they can use a single interface across more stores. For smaller or newer providers competing in this market, acquiring merchant network scale requires significant time and capital. The result is strong incumbent advantage in the core transactional product.

This competitive dynamic creates pressure to differentiate on dimensions other than network size. User experience — how easy and useful the platform's traveler-facing tools are — is one such dimension. A conversational assistant that helps travelers discover stores, plan their itinerary around tax-free shopping, and find products matching their preferences is a feature that large incumbents have not systematically deployed. The technical capability to build such a system — combining real transaction data, a scraped product catalog, and modern language model APIs — has only recently become accessible at reasonable cost, which means the window for differentiation on this dimension is currently open.

The specific design choice in this thesis — positioning the conversational assistant as a discovery and navigation tool rather than a pure transaction tool — directly supports differentiation strategy. A traveler using the assistant is engaging with the platform's ecosystem in a way that builds familiarity and preference; even if the underlying refund mechanism is commoditized, the assistant's utility creates switching costs that a pure transaction provider cannot match.

### 2.9 Synthesis

The literature reviewed above points toward a specific hybrid architecture for the tourist shopping recommendation problem. Content-based filtering using multimodal embeddings is appropriate for product-level search because it requires no interaction history, handles multilingual content naturally, and benefits from the visual richness of product catalogs. Collaborative filtering is appropriate for store-level recommendations because it captures behavioral patterns that content alone cannot (a store's overall character is not fully described by any single product's visual appearance). Demographic augmentation addresses the cold-start limitation of CF by substituting segment-level behavioral patterns when individual history is absent. Conversational interaction addresses the preference elicitation challenge, allowing users to express needs in natural language and refine recommendations through dialogue rather than structured query forms.

Evaluation of such a system must go beyond accuracy: coverage ensures the system is not confined to a small popular-item bubble; diversity measures whether recommendation lists offer genuine choice; novelty checks whether the system is surfacing items beyond the obvious; and transparency — provided through the conversational layer — supports user trust. The temporal evaluation protocol is essential for model-based CF to avoid the data leakage problem that leave-one-out cross-validation introduces in this setting.

### 2.10 Summary

This chapter has reviewed the foundations and current state of research relevant to the thesis design. Recommendation systems for retail and tourism are well-established, with hybrid architectures outperforming single-approach methods. Cold-start and sparsity are known challenges with several mitigation strategies, including demographic features and conversational preference elicitation. Multimodal CLIP embeddings represent a recent and powerful approach to cross-modal, multilingual product retrieval that transfers well to specialized retail domains. The tax-free intermediary market structure creates concrete strategic motivation for the conversational assistant as a differentiation tool. The following chapters present the data, methodology, and evaluation that test these ideas empirically.

---

## Chapter 3: Data and Context

### 3.1 The Tax-Free Shopping Context

The data used in this thesis originates from a tax-free shopping platform operating in the northern Italian region encompassing Lake Como and the greater Milan metropolitan area. The platform processes VAT refund claims for non-EU resident travelers making purchases at participating stores. Its data infrastructure captures two types of information that are relevant for recommendation: invoice records that document completed transactions, and product catalog content scraped from partner merchant websites.

These two data sources serve fundamentally different purposes and were developed under different constraints. Invoice records use normalized description-based product identifiers generated at point of sale; product catalog entries use URL-based identifiers from the merchants' own e-commerce systems. The two identifier spaces do not overlap. A transaction for "MAGLIA" at a given store cannot be reliably linked to any specific product in the scraped catalog, because the same generic description ("maglia" means knitted top in Italian) appears at hundreds of stores selling entirely different products. This non-linkability is not a data quality failure — it is a structural feature of how tax-free invoice records are generated. Its implications for methodology are discussed in Chapter 4.

### 3.2 Transaction Data

Source and initial scope. Transaction data was accessed through a Databricks pipeline that processes raw invoice records. An initial extract covering January 2025 only produced 24,665 line items across 2,951 customers and 345 stores. This one-month window was insufficient for recommendation: 79.5% of customers had only a single transaction, leaving almost no co-shopping signal for collaborative filtering.

Expansion. The data was expanded to approximately two years of transaction history, yielding 754,338 total items, 109,056 invoices, 39,777 unique customers, and 776 unique stores. This expansion dramatically improved the signal available for collaborative filtering: the number of customers with at least two distinct store visits grew from approximately 590 (with one month) to 8,572 — a fourteen-fold increase. These 8,572 customers are the viable training population for store-level CF.

Data structure. Each invoice record contains: a customer identifier, a store identifier and name, a transaction date, an item description, a price, and currency. Item descriptions are free-text fields entered at the point of sale by store staff. They are frequently generic ("maglia", "pantalone", "giacca", "accesorio") and provide insufficient information for product-level cross-merchant linking.

Temporal split. For evaluation purposes, the dataset is divided at December 1, 2025:
- Training set: 675,867 transaction items before the split date
- Test set: 78,471 transaction items from December 1, 2025 onward
- Test cases for store-level evaluation: 1,708 customers who appear in both periods and visited at least one store in the test period that they had not visited in the training period

Data quality issues. Two data quality issues were identified and addressed. First, a duplicate line item bug caused some invoice records to appear multiple times with identical fields; this was resolved by using position-indexed composite keys to deduplicate line items within each invoice. Second, product descriptions were confirmed to be insufficient for product-level CF: the same description string appears across hundreds of different stores, making description-based item similarity meaningless for distinguishing one store's products from another's.

**Key statistics:**

| Metric | Value |
|--------|-------|
| Total customers | 39,777 |
| Total stores | 776 |
| Total invoices | 109,056 |
| Total transaction items (training) | 675,867 |
| Total transaction items (test) | 78,471 |
| Customer–store matrix sparsity | 99.81% |
| Average distinct store visits per customer | 1.42 |
| Customers with 2+ store visits | 8,572 |

### 3.3 Scraped Product Catalog

Overview. Two partner merchants provided the product catalog data through web scraping. The fashion retailer contributed 134 valid products with approximately 736 product images. The pet supply retailer contributed 500 valid products with approximately 2,040 product images. After filtering products with empty names or zero images, 634 valid products were retained for embedding and evaluation.

Fashion retailer scraper. Products were collected through category pagination on the merchant's e-commerce website, iterating across product category pages and extracting product names, descriptions, prices, image URLs, and canonical product URLs. URL slugs follow a consistent pattern that encodes product category as the first word segment, enabling category extraction for evaluation purposes (Section 5.3).

Pet supply retailer scraper. Products were collected through sitemap-driven discovery, parsing the merchant's XML sitemap to identify product pages and then extracting product information from each page. Product names in this catalog follow conventions that encode both the target animal species and the product type, enabling keyword-based category extraction.

Storage. Products are stored in a SQLite database with deterministic SHA-256 product identifiers derived from the string "merchant::url". This determinism ensures that re-running the scraper produces consistent identifiers across runs, allowing the embedding index to remain stable without re-indexing after incremental catalog updates.

**Product data schema:** product identifier, merchant name, product name, description, price, currency, product URL, and a list of image URLs. Products are associated with images in a separate table, with multiple images per product supported.

Store–product matrix. The scraped catalog was used to construct a store–product matrix for item-based CF. This matrix is sparse (99.83% empty) but richer in product variety than the transaction-level description data, covering 776 stores and 121,197 derived product descriptions across the full transaction history.

### 3.4 Customer Demographics

Customer demographic data was assembled from two sources: a customer information file containing date of birth, and the transaction data containing nationality and country of residency. These were joined on customer identifier, producing a demographic profile for each of the 39,777 customers. Three features were derived for use in recommendation models:

- *Age bin*: customers were binned into age groups (under 30, 30–45, 45–60, 60 and over) from date of birth
- *Tourist type*: customers were classified as domestic (Italian residents), cross-border (residents of countries bordering Italy: Switzerland, Austria, France, Slovenia, San Marino), or international (all other non-domestic)
- *Nationality*: the country of nationality as recorded on the refund claim

**Tourist type distribution:**

| Tourist Type | Count | Percentage |
|-------------|-------|------------|
| Cross-border | 28,855 | 72.5% |
| International | 10,463 | 26.3% |
| Domestic | 459 | 1.2% |
| **Total** | **39,777** | **100%** |

Cross-border tourists — primarily Swiss, Austrian, French, and Slovenian residents — constitute nearly three-quarters of the customer base. This distribution reflects the geographic reality of the Como and Milan area: it sits at the intersection of three national borders and draws heavily from neighboring countries. The dominance of cross-border shoppers has implications for demographic modeling, discussed in Chapters 4 and 7.

### 3.5 Data Challenges

Several data challenges shaped the system design in ways worth noting explicitly.

The catalog and transaction data cannot be linked. Invoice item descriptions cannot be reliably matched to scraped catalog products, so product-level CF based on transaction data is not feasible: the interaction signal (what customers bought) cannot be connected to the content signal (what products exist in the catalog). The architectural response — separating the product recommendation layer (CLIP, content-based) from the store recommendation layer (CF, behavior-based) — follows directly from this constraint.

Item descriptions in the transaction data are frequently too generic to carry useful semantic information. "Maglia", "pantalone", and "giacca" each appear at hundreds of stores, covering enormous variation in price, style, brand, and quality. Product-level CF on these descriptions would produce recommendations based on co-occurrence of generic category terms rather than genuine product similarity.

With 72.5% of customers being cross-border tourists who often make a single trip, most customers have only one or two interaction records. This forces the recommendation system to extract maximum signal from minimal behavioral data, favoring model-based CF (which generalizes across users) and demographic approaches (which require no individual history) over memory-based CF (which needs enough interactions to compute reliable pairwise similarity).

Product names and descriptions span Italian and Spanish, with some English. The fashion retailer's products are described primarily in Spanish; the pet supply retailer uses Italian. Standard text processing tools that assume a single language will produce degraded results. CLIP's multilingual training corpus (Section 2.7) addresses this at the embedding level.

---

## Chapter 4: Methodology

### 4.1 System Overview

The recommendation system is organized into four distinct layers, each addressing a different aspect of the tourist shopping challenge.

The content-based layer uses OpenCLIP multimodal embeddings to support product discovery. A user can express a product preference in natural language or by uploading an image; the system converts the query to a 512-dimensional embedding and returns the most similar products from the 634-product catalog by cosine similarity. This layer requires no interaction history and operates effectively from a traveler's first contact with the system.

The collaborative filtering layer supports store-level recommendations. Given a customer's transaction history, the system predicts which stores the customer is likely to visit next, using one of five model architectures: item-based CF, user-based CF, ALS, LightFM WARP, and LightFM Hybrid. These models are trained on the full two-year transaction dataset at startup and are served through the API without re-training at inference time.

The demographic layer provides three additional store recommendation models that incorporate nationality, age, and tourist type. These models are particularly relevant for cold-start users and are evaluated as alternatives to — not replacements for — the behavioral CF models.

The conversational layer uses GPT-4o-mini with structured tool calling as the user interface for the entire system. Rather than exposing raw search and recommendation endpoints directly to users, the LLM interprets natural language queries, calls the appropriate tools with structured parameters, and curates the results before presenting them. This architecture allows the system to handle the gap between what users say and what structured search queries can retrieve.

These layers are intentionally separated: the chatbot handles product discovery through the content-based layer, while store recommendations from the CF and demographic models are displayed on a dedicated UI section. This separation allows each layer to be evaluated independently and updated without affecting the others.

### 4.2 CLIP Multimodal Embedding Pipeline

Model selection. The embedding model is OpenCLIP ViT-B-32, pretrained on LAION-2B using the checkpoint `laion2b_s34b_b79k`. This model produces 512-dimensional embeddings in a shared image-text space. The LAION-2B training data provides broad visual and semantic coverage including multilingual content, which is essential for a catalog with Italian and Spanish product names. The 512-dimensional space is compact enough for brute-force nearest-neighbor search at a catalog size of 634 products, avoiding the need for approximate indexing.

The alternative ViT-L-14 architecture (also available in OpenCLIP) produces 768-dimensional embeddings and achieves marginally higher zero-shot accuracy on standard benchmarks such as ImageNet. However, at a catalog scale of 634 products, the representational capacity of ViT-B-32 is already more than sufficient: the evaluation reports HR@10 = 0.956, meaning the model correctly ranks a category-relevant product in the top 10 for 95.6% of queries, leaving little headroom that a larger model could exploit. The 50% increase in embedding dimensionality (512 to 768) would increase the catalog matrix memory from approximately 1.3 MB to 2.0 MB and slow down batch encoding, with no practical benefit at this scale. Furthermore, Cherti et al. (2023) demonstrate that the ViT-B-32 checkpoint trained on LAION-2B substantially outperforms the original OpenAI CLIP ViT-B-32 checkpoint (trained on only 400 million pairs) on downstream transfer tasks, meaning the LAION-2B pretraining already closes much of the gap between architectures. The compute-accuracy trade-off therefore favors ViT-B-32 for this deployment context.

Text representation. Each product is represented as a text string combining its name and the first 200 characters of its description: "{name}. {description[:200]}". This format places the most semantically specific information (the product name) first, followed by descriptive context. The 200-character description truncation keeps the representation within CLIP's 77-token context window while retaining the most informative content.

Image encoding. Each product image is preprocessed with CLIP's standard transforms: resize to 224×224, center crop, and normalization with ImageNet mean and standard deviation. The preprocessed image is passed through the ViT-B-32 image encoder to produce a 512-dimensional embedding, which is then L2-normalized. For products with multiple images, the per-image embeddings are mean-pooled to produce a single product representation, then L2-normalized again.

The final product embedding is the mean of the image and text embeddings, then L2-normalized:

fused_embedding = L2-normalize( (mean_image_embedding + text_embedding) / 2 )

This equal-weight fusion gives equal importance to visual and textual signals. Because both component embeddings are already L2-normalized before fusion, neither dominates the combined representation by scale alone. The resulting embedding lies on the unit hypersphere, making cosine similarity equivalent to the dot product — relevant for efficient batch similarity computation.

At query time, the query is encoded using the same model and transforms, L2-normalized, and compared against all 634 catalog embeddings via matrix multiplication. The top-k results by cosine similarity are returned. No approximate indexing is used; at 634 products, brute-force matrix multiplication is sub-millisecond and avoids the OpenMP threading conflicts that FAISS introduces on macOS.

Before building the full pipeline, the model was validated on a small qualitative test. The text query "black converse high top sneaker with white sole" was compared against four product images: a black Converse high-top, a red Converse high-top, a generic black shoe, and red heels. Cosine similarities were: black Converse (0.3315), red Converse (0.2729), generic black shoe (0.2001), red heels (−0.0041). The ranking matches expected visual and semantic similarity, confirming that the model captures both color and style correctly.

### 4.3 Memory-Based Collaborative Filtering

#### 4.3.1 Why Store-Level?

Product-level CF on transaction data is not feasible for the reasons established in Chapter 3: generic item descriptions, non-linkability to the scraped catalog, and the prevalence of single-occurrence products (65% of transaction-level product descriptions appear only once). Aggregating to the store level creates a more stable and interpretable signal: a store's full product mix across all its transactions reveals its character as a retail entity, even when individual product descriptions are noisy.

Store-level CF also aligns better with the traveler's decision problem. A tourist planning their day is deciding which stores to visit, not which specific products to purchase. Store recommendations are directly actionable and do not require the user to already know what they want.

#### 4.3.2 Item-Based CF (Store Similarity)

The item-based CF model represents each store as a vector in product-description space. The interaction matrix has dimensions 776 stores × 121,197 unique derived product descriptions, stored as a sparse CSR matrix. Each cell contains the number of times a given product description appears in transactions at a given store. Store-to-store cosine similarity is then computed from these vectors.

The interpretation is that two stores are similar if they sell similar products in similar proportions. A boutique that frequently sells "vestido negro", "chaqueta", and "pantalon" will be similar to another boutique with the same transaction profile — regardless of brand, price level, or physical location. Given a query store, the model returns the most similar stores, excluding the query store itself and any store sharing the same (merchant name, city) combination to avoid trivially recommending branches of the same retail chain.

#### 4.3.3 User-Based CF (Co-Shopping Patterns)

The user-based CF model represents each customer as a vector in store space. The interaction matrix has dimensions 39,777 customers × 776 stores. Each cell records the number of visits the customer made to that store in the training period.

At recommendation time, the system finds the 20 most similar customers to the query user by cosine similarity of store visit vectors. It then aggregates the visit counts of those neighbors for each unvisited store, weighted by similarity, and normalizes the resulting scores. Deduplication removes stores that share a (merchant name, city) combination with already-visited stores. The top-k stores by normalized score are returned.

The key limitation of this approach in the current dataset is the sparsity of individual user vectors. A customer with one store visit has a vector with a single nonzero entry; cosine similarity between two such vectors is either 0 (if they visited different stores) or 1 (if they visited the same store). This binary behavior makes neighborhood-based similarity unreliable and is the primary reason user-based CF performs poorly in the evaluation results.

### 4.4 Model-Based Collaborative Filtering

#### 4.4.1 Interaction Matrix with Recency Weighting

All three model-based approaches use a recency-weighted customer–store interaction matrix. Raw visit counts are adjusted by an exponential decay function:

weight = visits × exp(−days_since_last_visit / 180)

The 180-day half-life constant means a visit from six months ago retains approximately 37% of its weight relative to a visit today. This captures the intuition that recent visits are more informative about current preferences than visits from over a year ago. Visit counts (not total spend) are used as the base signal; using spend would bias recommendations toward stores selling expensive goods, which does not necessarily reflect preference intensity.

The choice of 180 days is grounded in the seasonal structure of Como and Milan tourism. The primary tourist seasons in the Como lake district are summer (June–September) and winter (December–March), each lasting roughly four to five months, with a shoulder period in between. A 180-day decay constant means that a visit from the immediately preceding season retains meaningful weight (~37%), while a visit from two seasons ago (approximately one year) decays to roughly 14% and is treated as weak evidence. This aligns with the intuition that a tourist who visited a particular store during last summer's trip is likely to have similar preferences on a return visit, while a visit from two years prior is less predictive. Ding and Li (2005) provide theoretical and empirical support for time-weighted collaborative filtering, showing that recency-weighted interaction matrices produce more accurate recommendations than uniform weighting across a range of collaborative filtering methods. The specific constant was not grid-searched to optimize validation HR; it was set a priori based on the seasonal tourism calendar and is acknowledged as a design choice that could be refined with more extensive hyperparameter search.

#### 4.4.2 ALS (Alternating Least Squares)

ALS is a matrix factorization method for implicit feedback datasets introduced by Hu et al. (2008). Unlike SVD-based methods designed for explicit ratings, ALS treats all interactions as positive implicit feedback and learns user and item latent factors by minimizing a weighted reconstruction loss. The confidence weight for each observed interaction is proportional to the interaction strength (here, the recency-weighted visit count), while unobserved interactions are assigned uniform low confidence. This asymmetry — high confidence for observed, low confidence for unobserved — is the key distinction from standard matrix factorization and makes ALS suitable for implicit feedback data where absence of interaction does not imply dislike.

The model is configured with 64 latent factors, 30 iterations, and regularization parameter λ = 0.01. Recommendations are generated by computing the dot product between the user's learned factor vector and all item factor vectors, returning the top-k stores by predicted score.

#### 4.4.3 LightFM WARP

LightFM (Kula, 2015) is a hybrid recommendation model that natively supports both interaction data and item/user side features through a shared embedding space. The WARP (Weighted Approximate-Rank Pairwise) loss function directly optimizes for top-k ranking quality: during training, it samples negative items for each positive interaction and applies larger gradient updates when negatives are ranked higher than the positive item. This focuses training effort on the most consequential ranking errors — cases where the model is mistakenly placing a non-visited store above a visited one — which is directly aligned with the top-k evaluation objective.

The WARP loss is particularly well-suited to implicit feedback datasets where the goal is accurate top-k ranking rather than accurate prediction of the full interaction matrix. Rendle et al. (2009) show that pairwise ranking losses generally outperform pointwise reconstruction losses for implicit feedback recommendation, and the WARP variant's adaptive sampling concentrates this advantage on the high-rank positions where it matters most for users.

Configuration: 64 latent components, learning rate 0.05, 30 epochs, 4 threads.

#### 4.4.4 LightFM Hybrid (WARP with Side Features)

This model extends LightFM WARP by incorporating store-level item features and user behavior features. The LightFM model learns separate embeddings for each feature, and the effective user and item representations are sums of their feature embeddings — a form of feature hashing that allows the model to generalize to users or items with overlapping features.

Store (item) features encode: geographic city (one-hot), store size bin (small, medium, or large based on product count), and price bin (low, mid, or high based on median transaction price). Merchant name is deliberately excluded from store features: including it caused the model to over-index on within-merchant recommendations, recommending all stores from the same merchant chain regardless of individual store characteristics.

User features encode behavioral patterns from the training period: visit frequency bin (single, casual, regular, or power user based on visit count), spend level bin (low, mid, or high at P33 = 290 EUR and P66 = 890 EUR), and primary shopping city (the city where the customer has the most store visits).

Configuration: 128 latent components, learning rate 0.01, 50 epochs. The higher component count and slower learning rate reflect the larger feature space.

### 4.5 Demographic Recommenders

#### 4.5.1 Motivation

Behavioral CF models require at least some interaction history to produce personalized recommendations. For customers with zero history — new tourists arriving at a store for the first time — demographic features provide an alternative signal. The tax-free context makes nationality particularly informative: cross-border shoppers from Switzerland may systematically prefer different store categories than international tourists from Japan or the United States, reflecting differences in shopping behavior, product preferences, and travel purpose.

Three demographic models are evaluated, spanning a range from pure segment-based popularity to fully hybrid neural approaches.

#### 4.5.2 Demographic Popularity (Model 11)

Demographic Popularity is a segment-based popularity model. For each customer, the model identifies stores that were most frequently visited by customers in the same demographic segment during the training period. Segment is defined as the combination of nationality, age bin, and tourist type.

When a segment has fewer than 20 customers — which occurs for rare nationality combinations — a fallback hierarchy is applied: first drop tourist type (segment = nationality × age bin), then drop age bin (segment = nationality only), then fall back to global popularity across all customers. This hierarchy ensures that every customer receives a recommendation, even if their nationality-age-tourist_type combination is represented by only one or two customers.

#### 4.5.3 LightFM Demo (Model 12)

LightFM Demo uses LightFM WARP with demographic-only user features: nationality, age bin, and tourist type encoded as sparse indicator vectors. No behavioral features are included. The model learns to associate demographic profiles with store preferences purely from the co-occurrence of demographic group membership and store visits in the training data.

This model isolates the contribution of demographic signals from behavioral patterns: any performance gain over pure global popularity must come from demographic variation in store preferences, not from individual-level behavioral history. Configuration: 64 components, learning rate 0.05, 30 epochs.

#### 4.5.4 LightFM Full Hybrid (Model 13)

LightFM Full Hybrid combines all available feature types: behavioral user features (visit frequency, spend level, primary city), demographic user features (nationality, age bin, tourist type), and store item features (geographic region encoded as six regional clusters — lake_como, milan_metro, alpine_north, northeast, central, south — plus store size bin and price bin). This is the most feature-rich model evaluated.

The geographic region encoding replaces the city-level encoding used in LightFM Hybrid with higher-level regional clusters. This reduces the feature dimensionality while capturing the broad geographic variation in store offerings across the northern Italian region.

Configuration: 64 components, learning rate 0.05, 30 epochs.

#### 4.5.5 Ethical Considerations

Using nationality and age as recommendation features is a form of statistical discrimination: customers from the same national background receive the same baseline recommendations regardless of their individual preferences. In commercial deployment, this approach raises fairness concerns. A customer from China and a customer from Switzerland visiting the same store would receive different baseline recommendations based on their nationality, potentially creating a two-tier experience that correlates with national origin.

This thesis evaluates demographic models in a research context where the primary question is whether demographic signals carry predictive power. The quantitative results determine whether demographic features improve recommendation quality and, if so, whether the improvement justifies the fairness cost. The results in Chapter 6 show that demographic models do not meaningfully outperform behavioral CF models, which suggests that deploying demographics as the primary recommendation signal would impose fairness costs without corresponding accuracy gains.

In any commercial deployment, demographic-based personalization should require explicit user consent, transparent disclosure that nationality is used in recommendations, and mechanisms for users to opt out of demographic-based filtering.

### 4.6 Conversational AI Layer

The conversational component uses GPT-4o-mini with OpenAI's tool-calling interface as the control layer for product discovery. The LLM does not directly serve recommendations; it interprets user intent, formulates structured requests to the backend tools, and curates the results before presenting them.

The tools exposed to the LLM are:

- *search_products*: CLIP text search against the 634-product catalog, with optional price range and merchant filters
- *search_by_image*: CLIP image search using a user-uploaded image as the query
- *select_products*: takes a list of search results and returns the subset the LLM considers genuinely relevant to the user's stated need
- *ask_preferences*: presents structured follow-up questions to elicit preferences when the initial query is underspecified
- *explain_recommendation*: generates a plain-language explanation of why a specific product was recommended
- *find_nearby_stores*: returns stores from the CF recommendation list, optionally sorted by proximity to a given location

Interaction flow. A user sends a free-text message or uploads an image. The LLM receives the message and the conversation history, then decides whether to ask a clarifying question (if intent is ambiguous) or proceed to search. If searching, it calls search_products or search_by_image with appropriate parameters. The search results are then passed to select_products, which filters out results the LLM considers poor matches. The filtered results are sent to the frontend as a product carousel. Users can refine the recommendations — "something cheaper", "I want it for a medium-sized dog" — and the LLM interprets these as new constraints for a follow-up search.

**The select_products step** is a critical quality control mechanism. Without it, the system would always display a fixed number of search results regardless of their relevance to the specific query. By routing results through the LLM before display, the system surfaces only results that the LLM considers contextually appropriate. A minimum cosine similarity threshold of 0.10 pre-filters very poor matches before the LLM sees them, ensuring that the LLM's selection task starts from a set of plausible candidates.

Implementation. The tool-calling loop runs on the FastAPI backend. The OpenAI API key is held server-side and never exposed to the browser. Each tool call generates a structured JSON payload that the backend executes, with results returned to the LLM in the next message turn. The full conversation history is maintained in the session, allowing the LLM to track what has been shown and avoid repetition.

### 4.7 System Architecture

The complete system consists of three main components.

The backend is a FastAPI application that loads all models at startup: the OpenCLIP model and weights, the 634×512 embedding matrix, product metadata, the customer–store interaction matrix, and the trained ALS and LightFM model weights. CF and LightFM models are trained from the SQLite transaction database on first startup, taking approximately three seconds on Apple Silicon. Subsequent startups load pre-trained weights from disk, and the backend exposes REST endpoints for product search, store recommendation, conversation, and map data.

The frontend is a React application that communicates with the backend via REST. The UI includes a landing page, a conversational recommender page (chat interface, product carousel, and store map), a browsing page (product grid with filters), and an evaluation dashboard showing model comparison visualizations. Product carousels and map overlays render inline within the conversation.

Product catalog and transaction data are stored in SQLite. The CLIP embedding matrix is stored as a NumPy binary file (634×512 float32), and product metadata in Parquet format for efficient columnar access. Both are loaded into memory at startup for low-latency search.

---

## Chapter 5: Evaluation Design

### 5.1 Philosophy

Product-level and store-level recommendations operate under fundamentally different assumptions, data conditions, and evaluation constraints. Applying a single evaluation protocol to both would obscure the distinct technical challenges at each level. This chapter describes the two protocols and the metrics that apply to each.

### 5.2 Store-Level: Temporal Train/Test Split

Motivation and the leakage problem. An earlier version of the system used leave-one-out cross-validation for store-level evaluation: for each test user, one store was held out, the model was evaluated on whether it ranked the held-out store highly, and the procedure was repeated across users. This protocol produces an apparent evaluation but has a critical flaw for model-based CF (ALS, LightFM): during training, the model sees the full interaction matrix including the held-out interaction. The model's learned latent representations therefore already encode information about the user–store relationship being tested. The result is artificially inflated metrics — near-perfect hit rates that do not reflect the model's actual ability to generalize to new data.

Temporal split. The temporal split protocol eliminates this leakage by ensuring that no test-period interaction is visible during training:
- Training set: all transactions before December 1, 2025
- Test set: all transactions from December 1, 2025 onward
- All customer–store interaction matrices, model weights, and similarity matrices are computed from training-period data only

Test cases. Not all customers in the test period are informative test cases. Customers who appear only in the test period (with no training history) cannot be evaluated for personalized recommendation — there is nothing to personalize from. Customers who visit only stores they already visited in training are not testing discovery. The evaluation therefore focuses on 1,708 customers who appear in both periods and who visited at least one store in the test period that was not in their training history. These are the customers for whom the recommendation task — predicting future new store visits — is well-defined.

Evaluation task. For each of the 1,708 test customers, the system generates k recommendations using each model (trained on data through November 30, 2025). The recommendations are evaluated against the set of new stores the customer actually visited during the test period.

### 5.3 Product-Level: Self-Retrieval with Category Relevance

Protocol. For each of the 634 products in the catalog, the product's own CLIP embedding is used as the query, the product itself is excluded from the result set, and the top-k most similar products are retrieved. The evaluation measures whether products from the same semantic category appear in the top-k results.

**Why not same-merchant relevance?** Using all products from the same merchant as relevant would be too coarse. The pet supply retailer's 500 products span food, toys, grooming, health, and accessories for cats, dogs, and small animals; treating all 499 other pet products as equally relevant to a query about cat food inflates recall and renders precision meaningless. Category-based relevance provides a finer-grained test of whether the embedding captures genuine semantic similarity within the catalog.

Category extraction. Two different extraction methods are used for the two merchants, reflecting their different catalog structures.

For the fashion retailer: the product URL slug encodes category information as the first word segment before a product code suffix (e.g., "-261TP3083_00059_20"). Stripping the product code and taking the first word yields the category. This method successfully categorizes 133 of 134 products (99.3%) across 16 categories: vestido (34 products), jersey (25), pantalon (20), chaqueta (10), and 12 smaller categories.

For the pet supply retailer: keyword matching is applied to product names, identifying the animal type (cat, dog, small animal) and the product type (food, snack, toy, grooming, health, accessories, litter). The category is the cross-product of these two dimensions (e.g., "cat-food", "dog-toy"). This method categorizes 428 of 500 products (85.6%).

In total, 561 of 634 products (88.5%) receive a category-based relevance judgment. The remaining 11.5% fall back to same-merchant relevance — all other products from the same merchant are treated as relevant. This fallback is conservative and affects a small minority of products.

### 5.4 Metrics

Seven metrics are computed across both evaluation levels where applicable. Let R denote the set of relevant items, L_k the top-k recommendation list, and |·| denote set cardinality.

**Hit Rate@K (HR@K).** Equals 1 if at least one relevant item appears in L_k, and 0 otherwise. HR@K measures whether the system succeeds in surfacing at least one relevant recommendation. It is the weakest accuracy criterion — a single relevant item at position k qualifies — but is appropriate in settings where even one good recommendation is useful. (Deshpande and Karypis, 2004)

**Precision@K.** Defined as |R ∩ L_k| / k, the fraction of recommended items that are relevant. Precision penalizes lists that include many irrelevant items alongside a few relevant ones. (Manning et al., 2008)

**Recall@K.** Defined as |R ∩ L_k| / |R|, the fraction of all relevant items that appear in L_k. For users with many relevant items (e.g., a customer who visited 20 stores in the test period), achieving high recall requires surfacing many of them. (Manning et al., 2008)

**nDCG@K.** Normalized Discounted Cumulative Gain with binary relevance and log₂ discount. The discounted cumulative gain is DCG@K = Σ_{i=1}^{k} rel_i / log₂(i + 1), where rel_i ∈ {0, 1} indicates whether the item at position i is relevant. nDCG@K normalizes DCG@K by the ideal DCG achieved when all relevant items appear at the top of the list: nDCG@K = DCG@K / IDCG@K. This penalizes models that rank relevant items lower in the list, making it sensitive to ranking quality rather than just set membership. (Järvelin and Kekäläinen, 2002)

**Coverage.** The fraction of the full item catalog (all 634 products or all 776 stores) that appears in at least one recommendation list across all test cases. Low coverage indicates that the model concentrates recommendations on a small subset of popular items. (Adomavicius and Kwon, 2012)

**Diversity (ILD — Intra-List Diversity).** The average pairwise dissimilarity among the items in each recommendation list, averaged across all test cases. Dissimilarity between items i and j is defined as 1 − cos(e_i, e_j), where e_i and e_j are the CLIP (for products) or store profile (for stores) embeddings. Higher ILD indicates that the recommended items are diverse; a value of 0 means all recommended items are identical; a value of 1 means all pairs are orthogonal. (Ziegler et al., 2005)

**Novelty (product-level only).** The mean self-information of recommended items based on their recommendation frequency across all queries:

novelty = mean_{i ∈ recs}[ −log₂(freq(i) / N_queries) ]

where freq(i) is the number of queries for which item i appears in the top-k list, and N_queries is the total number of evaluation queries. An item recommended in every query has freq(i)/N_queries ≈ 1, yielding novelty ≈ 0. An item recommended in only 1% of queries has −log₂(0.01) ≈ 6.64 bits of novelty. This formulation, from Vargas and Castells (2011), measures the degree to which the recommendation distribution is concentrated on popular items versus spread across the full catalog.

### 5.5 Baselines

Two baselines are used at each recommendation level.

The random baseline samples k items uniformly at random from the full item set at query time, excluding items already known to the user (previously visited stores or the query product itself). It establishes the lower bound for accuracy metrics and the upper bound for diversity, since a random list is maximally diverse by construction. Coverage is 1.0 by the law of large numbers.

The popularity baseline recommends the most-visited stores during the training period (for store evaluation) or the most frequently co-retrieved products in the evaluation distribution (for product evaluation), excluding previously seen items for each user. It captures the non-personalized tendency for popular stores to attract many visitors; any model claiming to be personalized must outperform it. The popularity baseline's near-zero diversity and near-zero novelty make it a useful reference: strong popularity baseline performance indicates a dataset where global popularity dominates individual preference.

### 5.6 Embedding Ablation Protocol

To empirically validate the fusion design choice in Section 4.2, three embedding variants are compared under the same self-retrieval evaluation protocol: (1) image-only — mean-pooled image embeddings, with text embedding as fallback for products with no associated image; (2) text-only — the text embedding alone, ignoring image data; and (3) fused — the equal-weight mean of image and text embeddings, as deployed in the system. All three variants use the same ViT-B-32 backbone. The variants differ only in which modality inputs are used; the similarity computation, relevance judgment, and metrics are identical to Section 5.3.

The ablation answers the question: does the fusion of image and text embeddings outperform either modality alone, and by how much? A substantial gap would justify the added complexity of computing and fusing two separate embeddings. A negligible gap would suggest the simpler text-only approach is sufficient.

### 5.7 Text-Query Evaluation Protocol

The self-retrieval evaluation (Section 5.3) tests embedding consistency — whether a product's embedding is close to other products in the same category — but does not test whether the system correctly responds to natural language queries from real users. A complementary text-query evaluation was designed using 30 curated queries: 15 fashion queries (e.g., "summer floral dress", "slim fit black trousers", "knit oversized jumper") and 15 pet supply queries (e.g., "dry cat food for indoor cats", "dog harness for large breeds", "cat litter clumping"). Queries were chosen to span the major product categories in both catalogs and to reflect realistic user intent.

For each query, the system retrieves the top-k products using the live `search_by_text` pipeline (identical to the deployed chatbot search). Two binary metrics are recorded per query: merchant hit rate (1 if at least one returned product belongs to the correct merchant category) and category hit rate (1 if at least one returned product belongs to the queried product category, as defined by the category extraction logic in Section 5.3). Hit rates are averaged across all 30 queries at k = 5 and k = 10.

This protocol validates the end-to-end text-to-product retrieval path — including tokenization, text encoding, and cosine similarity search — in addition to the embedding quality captured by self-retrieval.

### 5.8 Conversational Layer Evaluation Protocol

The conversational layer (Section 4.6) is evaluated through structured scenario testing. Twenty-five scenarios were designed to cover six behaviorally distinct categories: simple product search (single-turn queries with a clear product intent), budget-constrained search (queries specifying a price range), multi-turn dialogue (a vague first message followed by a clarifying second message), cross-merchant queries (requests spanning both merchants in one conversation), store location queries (requests for physical store information), and edge cases (non-English input, out-of-scope requests).

For each scenario, the system is invoked programmatically using the same `chat()` function as the production API. Multi-turn scenarios are simulated correctly: the first user message is sent, the assistant response is captured, and the second user message is sent with the full conversation history including the assistant's first reply. This mirrors the actual user experience of a two-turn dialogue.

A scenario is marked as a success if either: (a) the system returns at least one product recommendation, or (b) for scenarios marked `clarification_acceptable` — five deliberately vague queries where asking for preferences is the correct behavior — the response contains a clarifying question (detected by the presence of "?" in the response text). The `clarification_acceptable` flag acknowledges that GPT-4o-mini exhibits two legitimate behaviors: immediately searching for specific queries, and asking for clarifying information for vague ones. Binary success is used rather than a graded score to keep the evaluation simple and reproducible.

### 5.9 Hyperparameter Optimization Protocol

Store-level model-based approaches (ALS, LightFM WARP, LightFM Hybrid) depend on continuous hyperparameters whose default values may not be optimal for this specific dataset. To assess whether the default-parameter results in Table 5 are robust and to characterize the sensitivity of performance to hyperparameter choice, Bayesian optimization was conducted using Optuna (Akiba et al., 2019) with 40 trials per model.

A nested temporal validation split was used to avoid any leakage from the test set. The optimization procedure is:

1. Define a validation period (2025-10-01 to 2025-11-30) within the training data.
2. Train each candidate hyperparameter configuration on data prior to 2025-10-01.
3. Evaluate HR@10 on the validation set (customers with new visits in the validation period not seen in training).
4. After 40 trials, select the hyperparameter configuration with the highest validation HR@10.
5. Retrain the winning configuration on the full training set (all data prior to 2025-12-01).
6. Evaluate the retrained model on the held-out test set (2025-12-01 onward).

The test set is never used during optimization. The search ranges were: ALS — factors ∈ [16, 256] (log scale), regularization ∈ [10⁻⁴, 1.0] (log scale), iterations ∈ [10, 50]; LightFM WARP and Hybrid — no_components ∈ [16, 256] (log scale), learning_rate ∈ [10⁻³, 0.5] (log scale), epochs ∈ [10, 100].

---

## Chapter 6: Results

### 6.1 Product-Level Results

The following table presents the evaluation results for the three product-level models at k = 10.

**Table 1. Product-level evaluation results (k = 10)**

| Model | HR@10 | nDCG@10 | Precision@10 | Recall@10 | Coverage | Diversity | Novelty |
|-------|-------|---------|--------------|-----------|----------|-----------|---------|
| CLIP Content-Based | 0.956 | 0.693 | 0.620 | 0.247 | 0.981 | 0.208 | 5.662 |
| Random Baseline | 0.435 | 0.131 | 0.129 | 0.015 | 1.000 | 1.000 | 5.918 |
| Popularity Baseline | 0.456 | 0.157 | 0.157 | 0.017 | 0.017 | 0.000 | 0.012 |

CLIP achieves HR@10 = 0.956, meaning that for 95.6% of the 634 query products, at least one same-category product appears in the top ten results. This compares to 0.435 for the random baseline and 0.456 for the popularity baseline. The improvement in nDCG is even larger: CLIP achieves 0.693 against 0.131 (random) and 0.157 (popularity), indicating that relevant items appear near the top of the list rather than merely somewhere in the top ten. Precision@10 = 0.620 means that on average 6.2 of the 10 recommended products are from the same category as the query — a strong result for a catalog spanning two merchants with entirely different product domains.

Coverage is 0.981, meaning 98.1% of the 634-product catalog appears in at least one recommendation list across all queries. The popularity baseline reaches only 1.7% catalog coverage, confirming that it always recommends the same products regardless of query.

On novelty, CLIP scores 5.662. The popularity baseline scores 0.012 — essentially zero — because the same products appear in nearly every list by design. The random baseline scores 5.918, slightly above CLIP, because random selection is by construction spread uniformly across the catalog. The small gap between CLIP (5.662) and random (5.918) confirms that CLIP recommendations are distributed across the catalog rather than concentrated on a popular subset.

CLIP achieves an ILD of 0.208, well below the random baseline (1.000). This is expected: within-category retrieval naturally returns visually and semantically similar items, which have high cosine similarity and therefore low pairwise dissimilarity. A list of ten similar dresses has low diversity by design — it is doing its job. Diversity is more informative for store-level recommendations, where it measures whether the system recommends a varied set of store types.

At k = 5, CLIP achieves HR@5 = 0.934 and nDCG@5 = 0.748. The slight decrease in hit rate and increase in nDCG follow the expected pattern: smaller k leaves fewer positions for relevant items to appear, but those that do appear cluster closer to position 1. The LightFM WARP store model moves from HR@5 = 0.168 to HR@10 = 0.248, a larger absolute gain, because store recommendations benefit more from a wider list given the sparser relevant signal.

#### 6.1.1 Ablation Study: Embedding Modality

To verify that combining image and text embeddings improves retrieval quality over either modality alone, an ablation study was conducted with three embedding variants: image-only (mean-pooled image embeddings, with text fallback for products lacking images), text-only (text embeddings only), and the fused combination used in the deployed system. All three variants use the same ViT-B-32 backbone and are evaluated with the same category-based relevance protocol at k = 10.

**Table 2. CLIP embedding ablation study (k = 10, category-based relevance)**

| Embedding Variant | HR@10 | nDCG@10 |
|-------------------|-------|---------|
| Fused (image + text) | 0.956 | 0.693 |
| Image-only | 0.935 | 0.577 |
| Text-only | 0.804 | 0.339 |

Fused embeddings achieve the highest performance on both metrics. Text-only embeddings perform substantially worse (HR@10 = 0.804, nDCG@10 = 0.339), confirming that visual information is important for product retrieval in a fashion and pet supplies catalog. Image-only embeddings perform well (HR@10 = 0.935) but fall short of the fused variant, demonstrating that text descriptions add useful disambiguation — products that look similar (two black cats on packaging, for example) can be distinguished by their names. The consistent ordering across both metrics confirms that the equal-weight fusion design is empirically justified for this catalog.

#### 6.1.2 Text-Query Evaluation

The self-retrieval evaluation uses product embeddings as queries, which tests the internal consistency of the embedding space but does not directly measure whether users can find products using natural language. A complementary evaluation was conducted using 30 curated natural language queries (15 fashion, 15 pet supplies) against the full catalog at k = 5 and k = 10. Two metrics were recorded: merchant hit rate (whether the top-k results contain at least one product from the correct merchant category) and category hit rate (whether the results contain at least one product from the queried category).

**Table 3. Text-query evaluation results (30 natural language queries)**

| Metric | k = 5 | k = 10 |
|--------|-------|--------|
| Merchant HR | 1.000 | 1.000 |
| Category HR | 0.767 | 0.800 |

Merchant routing is perfect at both k = 5 and k = 10: every query for a fashion product returns at least one fashion product, and every query for a pet supplies product returns at least one pet product. Category hit rate at k = 10 is 80% (24 of 30 queries). The 6 misses are concentrated in fine-grained sub-categories with sparse catalog representation: aquarium fish food, rabbit food, bird food, and cat scratching posts, all of which have very few products in the Arcaplanet catalog. The high merchant hit rate confirms that the shared embedding space correctly separates the two merchant domains without any explicit domain label; the sub-100% category hit rate reflects a genuine coverage limitation of the current catalog rather than an embedding failure.

#### 6.1.3 Conversational Layer Evaluation

The conversational layer — GPT-4o-mini with tool calling — was evaluated on 25 structured scenarios spanning six category types: simple product search, budget-constrained queries, multi-turn dialogue, cross-merchant queries, store location queries, and edge cases (non-English input, out-of-scope requests). Each scenario was evaluated on whether the system returned at least one relevant product or, for deliberately vague queries, whether it appropriately asked a clarifying question.

**Table 4. Conversational layer evaluation (25 scenarios)**

| Category | Scenarios | Success Rate |
|----------|-----------|-------------|
| Budget-constrained | 4 | 1.000 |
| Store location | 3 | 1.000 |
| Simple search | 8 | 0.625 |
| Edge cases | 2 | 0.500 |
| Multi-turn | 5 | 0.400 |
| Cross-merchant | 3 | 0.333 |
| **Overall** | **25** | **0.680** |

Budget-constrained and store location scenarios achieve perfect scores, confirming that the price filtering and `find_nearby_stores` tool integrations function correctly. Simple search achieves 62.5%, with failures primarily on highly specific queries where the catalog does not contain the requested product (e.g., a specific niche pet food brand). Multi-turn scenarios achieve 40%, a lower rate attributable to cases where the model fails to carry context across turns or issues a second search that overrides the context from the first. Cross-merchant queries achieve 33%, the weakest category, reflecting difficulty in simultaneously satisfying constraints from two independent product domains in a single response. The overall success rate of 68% should be interpreted as a lower bound: the evaluation protocol is binary (success or failure per scenario) and does not credit partial successes, such as a response that returns three of five requested product types correctly.

### 6.2 Store-Level Results

The following table presents the evaluation results for all 13 store-level models at k = 10, ordered by HR@10.

**Table 5. Store-level evaluation results (k = 10)**

| Model | HR@10 | nDCG@10 | Precision@10 | Recall@10 | Coverage | Diversity |
|-------|-------|---------|--------------|-----------|----------|-----------|
| LightFM WARP | 0.248 | 0.126 | 0.026 | 0.227 | 0.577 | 0.898 |
| LightFM Full Hybrid | 0.230 | 0.116 | 0.024 | 0.211 | 0.180 | 0.216 |
| LightFM Hybrid | 0.214 | 0.107 | 0.022 | 0.191 | 0.245 | 0.219 |
| LightFM Demo | 0.212 | 0.098 | 0.022 | 0.191 | 0.132 | 0.450 |
| Demographic Popularity | 0.174 | 0.081 | 0.018 | 0.161 | 0.263 | 0.000 |
| ALS | 0.151 | 0.083 | 0.015 | 0.136 | 0.304 | 0.813 |
| Popularity Baseline | 0.155 | 0.071 | 0.016 | 0.140 | 0.019 | 0.000 |
| Item-Based CF | 0.119 | 0.070 | 0.012 | 0.110 | 0.575 | 0.593 |
| User-Based CF | 0.067 | 0.040 | 0.013 | 0.060 | 0.429 | 0.973 |
| Random Baseline | 0.019 | 0.008 | 0.002 | 0.017 | 1.000 | 1.000 |

#### 6.2.1 Model-Based versus Memory-Based Collaborative Filtering

LightFM WARP (HR@10 = 0.248) clearly outperforms both memory-based CF models: Item-Based CF (HR@10 = 0.119) and User-Based CF (HR@10 = 0.067). ALS (HR@10 = 0.151) also surpasses both memory-based approaches. The performance gap reflects the fundamental limitation of memory-based CF under extreme sparsity: cosine similarity between near-empty interaction vectors is unreliable, producing noisy neighbor estimates that do not translate into accurate recommendations.

Model-based approaches — ALS and LightFM — learn dense latent representations that generalize across the sparse matrix. Even users with a single store visit receive a recommendation based on the latent factors learned from the aggregate behavior of the thousands of customers who visited that same store. This interpolation capacity is the essential advantage of model-based CF when behavioral data is scarce.

User-Based CF performs worse than Item-Based CF (0.067 versus 0.119). This is consistent with the dataset's characteristics: with an average of 1.42 store visits per customer, user vectors are almost entirely empty, making user similarity unreliable. Store vectors, by contrast, are populated with all transaction product descriptions, providing richer content for similarity computation.

#### 6.2.2 LightFM WARP versus ALS

LightFM WARP (HR@10 = 0.248, nDCG@10 = 0.126) outperforms ALS (HR@10 = 0.151, nDCG@10 = 0.083) by a substantial margin. The WARP loss's direct optimization for top-k ranking is the most plausible explanation: ALS minimizes a weighted reconstruction loss across the entire interaction matrix, treating every user-store pair as a prediction target. WARP, by contrast, focuses training effort specifically on the high-rank positions that determine hit rate and nDCG. In a dataset where the goal is accurate top-10 recommendation, WARP's objective is more directly aligned with the evaluation criterion.

#### 6.2.3 LightFM WARP versus Hybrid Models

LightFM WARP (HR@10 = 0.248) outperforms both LightFM Hybrid (HR@10 = 0.214) and LightFM Full Hybrid (HR@10 = 0.230). Adding store and user side features to the LightFM model does not improve recommendation accuracy over the interaction-only WARP baseline.

One explanation is that the features available — city, size bin, price bin, visit frequency bin — do not carry additional discriminative information beyond what the interaction patterns already encode. A store's city and price level can be inferred implicitly from its interaction patterns (customers who visit it versus those who do not), so explicit feature encoding adds little. A second explanation is that the hybrid models use different hyperparameters (128 components for LightFM Hybrid, more epochs) that may not be optimal for this specific dataset. Systematic hyperparameter search might close the gap.

The coverage difference is striking: LightFM WARP covers 57.7% of the store catalog, while LightFM Full Hybrid covers only 18.0%. The hybrid models concentrate their recommendations on fewer stores, potentially because feature-based embeddings collapse stores with similar features into similar latent representations, reducing the diversity of recommended options.

#### 6.2.4 Demographic Models

LightFM Demo (HR@10 = 0.212) is competitive with LightFM Hybrid (HR@10 = 0.214) despite using only demographic features and no behavioral history. This is a notable finding: for this dataset, knowing a customer's nationality, age, and tourist type provides roughly as much predictive power as knowing their store visit history. One interpretation is that tourist type — particularly the cross-border/international distinction — captures much of the behavioral variation between customer segments. Cross-border shoppers from neighboring countries tend to visit different store categories than long-haul international travelers, and this systematic variation is captured by demographic segmentation.

Demographic Popularity (HR@10 = 0.174) outperforms the global Popularity Baseline (HR@10 = 0.155), confirming that segmenting popularity by demographic group provides a meaningful improvement over non-personalized popularity. The improvement is modest, but it is achievable with zero individual interaction history — making Demographic Popularity a practical cold-start solution.

LightFM Full Hybrid (HR@10 = 0.230) ranks second among all models, suggesting that combining behavioral and demographic signals does provide some benefit. However, the gain over pure WARP (0.248) is negative — LightFM Full Hybrid is 7.2 percentage points below LightFM WARP — indicating that the combination still does not improve over behavior alone.

#### 6.2.5 Diversity Analysis

The diversity results reveal an important pattern. LightFM WARP combines the highest hit rate (0.248) with high diversity (0.898), meaning its recommendation lists cover a variety of store types even while achieving accurate predictions. ALS also achieves high diversity (0.813) alongside its moderate hit rate (0.151).

Popularity Baseline and Demographic Popularity both have diversity of 0.000 — their recommendation lists are identical for all users within the same segment, with no intra-list variation. This is a structural property of popularity-based models, not a calibration failure.

User-Based CF achieves the highest diversity (0.973) but the worst hit rate (0.067) among non-random models. The diffuse, spread-out recommendations of User-Based CF cover many store types but do not concentrate on the stores the user will actually visit. This is the canonical accuracy-diversity tradeoff: the model maximizing diversity achieves it at the cost of relevance.

LightFM WARP appears to achieve a favorable position on this tradeoff: high accuracy and high diversity simultaneously. This is plausible when the learned latent space distributes stores across multiple directions corresponding to different store types — recommendations in that space are diverse by the geometry of the learned representations.

#### 6.2.6 Coverage

LightFM WARP covers 57.7% of the 776-store catalog across all 1,708 test users, meaning that more than half of all stores appear in at least one recommendation list. Item-Based CF achieves nearly identical coverage (57.5%) despite a much lower hit rate, suggesting it spreads recommendations broadly but without precision.

Popularity Baseline covers only 1.9% of stores — it recommends the same small set of globally popular stores to every user. Demographic Popularity improves this to 26.3%, as different segments receive different popular stores. LightFM Full Hybrid has lower coverage (18.0%) than LightFM WARP (57.7%), consistent with the feature-based collapse noted above.

#### 6.2.7 Hyperparameter Sensitivity

The store-level results in Table 5 were produced with default hyperparameters (ALS: 64 factors, λ = 0.01, 30 iterations; LightFM WARP: 64 components, lr = 0.05, 30 epochs). To assess whether the reported rankings are robust to hyperparameter choice, Bayesian hyperparameter optimization was conducted using Optuna with 40 trials per model. A nested temporal validation split was used: models were trained on data prior to 2025-10-01, validated on visits from 2025-10-01 to 2025-11-30, and the best configuration was then retrained on the full training set (prior to 2025-12-01) before final test evaluation.

**Table 6. Hyperparameter tuning results (Optuna, 40 trials, k = 10)**

| Model | Best Hyperparameters | Val HR@10 | Test HR@10 | Test nDCG@10 | Default Test HR@10 |
|-------|---------------------|-----------|------------|--------------|-------------------|
| ALS | factors=16, λ=0.107, iter=33 | 0.267 | 0.244 | 0.129 | 0.151 |
| LightFM WARP | components=242, lr=0.008, epochs=29 | 0.341 | 0.318 | 0.169 | 0.248 |
| LightFM Hybrid | components=184, lr=0.263, epochs=10 | 0.262 | 0.226 | 0.115 | 0.230 |

Several findings are notable. First, the model ranking is preserved: LightFM WARP remains the best model after tuning (HR@10 = 0.318 versus ALS at 0.244 and LightFM Hybrid at 0.226), confirming that the ranking in Table 2 is not an artifact of default hyperparameters. Second, tuning provides a meaningful improvement for LightFM WARP (+28% relative improvement from 0.248 to 0.318) and a significant improvement for ALS (+61% relative, from 0.151 to 0.244). Third, LightFM Hybrid shows minimal gain from tuning (0.230 to 0.226, within noise), suggesting that the performance gap between Hybrid and WARP reflects a structural difference — the available side features simply do not add discriminative information — rather than a suboptimal hyperparameter configuration.

The optimal ALS configuration uses only 16 latent factors, well below the default of 64. This is consistent with the extreme sparsity of the interaction matrix: with 99.81% missing values, a 16-dimensional latent space is already more expressive than the data density can reliably support. Overfitting risk increases with model capacity in sparse settings, and the Optuna search correctly identifies a lower-dimensional solution. The optimal LightFM WARP configuration uses 242 components — much higher than the default 64 — with a very low learning rate (0.008), suggesting that many iterations over a wide latent space are beneficial when the WARP objective is optimized carefully with small gradient steps.

---

## Chapter 7: Discussion

### 7.1 Content-Based Filtering as the Primary Recommendation Engine

The CLIP evaluation results confirm the central hypothesis that motivated the product recommendation architecture: multimodal embeddings trained on large-scale web data transfer effectively to specialized retail domains, even when the catalog is multilingual and cross-merchant. The HR@10 = 0.956 result under a conservative self-retrieval protocol indicates that the embedding space is highly consistent within product categories. When the model is queried with a dress embedding, it returns dresses; when queried with a cat food embedding, it returns cat food products — and this holds reliably enough that only 4.4% of queries fail to retrieve a same-category product in the top ten.

The cold-start advantage is the key practical benefit. CLIP requires no purchase history: a tourist who has never used the platform receives the same quality of product search as a repeat customer with a detailed transaction record. For an ecosystem where the median customer visits stores only once, this is not a minor edge case — it describes the majority of users. Content-based filtering that requires zero history is the only approach capable of serving the majority of the customer base effectively.

The system's principal limitation at the product level is catalog scope. Currently, only two merchants' products are embedded. The transaction data spans 776 stores across 286 merchants, but product-level content is available for only 2. A traveler interested in electronics, jewelry, sportswear, or regional food products cannot be served by the current content-based system. Expanding the scraping pipeline to additional merchants would directly improve the system's utility without requiring any changes to the embedding or search infrastructure — the pipeline is designed to accommodate new products incrementally.

### 7.2 Collaborative Filtering Under Extreme Sparsity

The store-level results must be understood in the context of the dataset's sparsity. With an average of 1.42 store visits per customer and a 99.81% empty interaction matrix, any model that significantly exceeds the random baseline is extracting meaningful signal from an extraordinarily constrained data environment. The random baseline achieves HR@10 = 0.019; LightFM WARP achieves HR@10 = 0.248, a thirteen-fold improvement. Against the more meaningful popularity baseline (HR@10 = 0.155), LightFM WARP shows a 60% relative improvement.

The performance gap between model-based and memory-based CF is expected given the data characteristics but is worth quantifying. User-Based CF (HR@10 = 0.067) performs worse than the popularity baseline (HR@10 = 0.155), meaning that computing cosine similarity between near-empty customer vectors and using those similarity estimates to weight store counts actually produces worse recommendations than simply recommending globally popular stores. This is a specific, practical demonstration of the sparsity failure mode: the similarity estimates are so unreliable that they introduce noise rather than signal.

Item-Based CF (HR@10 = 0.119) performs better than User-Based CF because store product-description vectors are denser than customer vectors. However, it still trails ALS (HR@10 = 0.151) and LightFM WARP (HR@10 = 0.248) by a wide margin. The latent factor models' ability to interpolate across the sparse matrix — assigning recommendations to users based on the aggregate patterns of many similar users' interactions, rather than attempting direct customer-to-customer similarity on near-empty vectors — is the mechanism of improvement.

The finding that LightFM WARP outperforms LightFM Hybrid raises a question about feature engineering. Adding store features (city, size, price) and user features (visit frequency, spend level, primary city) to the interaction model should, in principle, improve cold-start performance by providing signal for users with sparse history. The fact that it does not improve top-k accuracy over pure WARP in this dataset suggests that either the feature encodings are not sufficiently discriminative, or the interaction patterns themselves already encode the relevant feature variation implicitly. A store in central Milan attracts different customers than a store in a lakeside village; the interaction matrix captures this variation through who visits which stores, without requiring an explicit geographic feature. Finer-grained features — district-level location, merchant category taxonomy, store visit seasonality — might add incremental value.

### 7.3 Demographic Models and the Cold-Start Problem

The demographic results provide a nuanced picture of when nationality and tourist type are useful signals. Demographic Popularity (HR@10 = 0.174) outperforms global popularity (HR@10 = 0.155) without any individual behavioral history, confirming that demographic segmentation carries real predictive information. Cross-border shoppers from Switzerland do, on average, visit different stores than international tourists from China — and this systematic difference is exploitable for recommendation.

The near-equivalence of LightFM Demo (demographic-only, HR@10 = 0.212) and LightFM Hybrid (behavioral, HR@10 = 0.214) is the most striking demographic finding. For this dataset, knowing a customer's nationality and tourist type provides essentially the same information as knowing their store visit history. One interpretation is that tourist type segmentation is a coarse but effective proxy for the behavioral variation that interaction history captures directly. Cross-border shoppers from neighboring countries visit the region frequently and exhibit systematic shopping patterns (grocery, everyday fashion, accessories) that are predictable from their origin. International tourists from Asia or the Americas may be making a once-in-a-decade trip and focus on luxury and souvenir purchases. These systematic differences are captured by nationality segmentation, so behavioral history adds little beyond confirming what nationality already predicts.

The implication for cold-start handling is practically significant. A new user who has just registered on the platform with their nationality — which is required for VAT refund eligibility — can immediately receive demographic-model-quality recommendations (HR@10 = 0.212) before making a single purchase. After their first store visit, the system can switch to or blend with behavioral CF. This staged approach — demographic recommendations until sufficient behavioral history accumulates — is a natural deployment architecture for the tourist shopping context.

The ethical considerations raised in Section 4.5.5 bear on the deployment question. If demographic models do not meaningfully outperform behavioral models for users who have any history, the fairness cost of nationality-based recommendations is not offset by accuracy gains. Deploying demographic models only for genuinely cold-start users — those with zero interaction history — and switching to behavioral models as soon as history is available would minimize the exposure of nationality-based profiling while preserving its utility where it matters.

### 7.4 The Conversational Layer

The GPT-4o-mini tool-calling architecture addresses a limitation that ranking metrics cannot capture: the gap between a user's natural language expression of a need and the structured parameters that an information retrieval system requires. A tourist saying "I want something for my friend who loves the outdoors" does not know whether to search "hiking equipment" or "outdoor clothing", whether to filter by merchant, or what price range is appropriate. The LLM bridges this gap by decomposing the natural language intent into structured tool calls — and by asking a clarifying question if the intent is genuinely ambiguous.

The select_products tool is the most consequential architectural choice in the conversational layer. Its role is to apply world knowledge and contextual judgment that cosine similarity cannot. If a user asks for "a gift for a 10-year-old boy" and the CLIP search returns a mix of toys, clothes, and pet products, the LLM's select_products call can filter to the genuinely appropriate items. This elevates the system from a retrieval interface to a curation interface, which is a qualitatively different user experience.

The absence of a formal quantitative evaluation for the conversational component is an acknowledged gap. Turn-level metrics — number of clarification turns before a recommendation, task completion rate, user satisfaction scores — would provide a rigorous assessment of conversational quality. These require either user studies with real travelers or a labeled dialogue dataset. Neither was available for this thesis. The conversational behavior was validated qualitatively through manual testing across a range of query types, confirming that the system handles free-text product queries, image-based queries, follow-up refinements, and price-filtered searches correctly. A formal evaluation remains an important direction for future work.

### 7.5 Limitations

The content-based system currently covers only two merchants' products. The tax-free ecosystem encompasses hundreds of merchants across many product categories that are unsearchable through the product discovery interface, which limits the assistant's practical utility for travelers whose interests fall outside fashion and pet supplies — which is most travelers.

The test period covers December 2025 through February 2026, a single winter season. Shopping patterns in the northern Italian region likely differ across seasons: summer lake tourism generates different store visit patterns than winter ski tourism. A more robust evaluation would cover at least one full calendar year.

The product-level evaluation measures embedding quality — whether CLIP embeddings are consistent within categories — not whether recommended products are ones users would actually purchase. Embedding consistency is necessary but not sufficient for recommendation utility. A product could be consistently embedded in the right category but still be the wrong price, size, or style for the user. Click-through data or purchase data would be needed to validate real-world utility.

As noted above, the conversational component is evaluated qualitatively rather than quantitatively. This is a significant gap given that the conversational interface is the primary user-facing contribution of the system.

The model hyperparameters for ALS and LightFM were optimized using Bayesian search (Section 6.2.7), confirming that LightFM WARP is the best-performing model under both default and tuned settings. However, the full range of hyperparameter options for LightFM Hybrid — including alternative loss functions and feature encoding schemes — was not exhaustively explored, and further search may narrow the gap with WARP.

### 7.6 Future Directions

The highest-value near-term extension is scraping additional merchant catalogs and adding them to the CLIP index. The scraping infrastructure is already in place; the bottleneck is identifying which merchants have e-commerce sites amenable to scraping. Even partial expansion — ten additional merchants covering jewelry, sportswear, and regional food — would meaningfully expand the system's utility.

Within a single shopping trip, a traveler's store visits may follow a sequential trajectory that session-based models (GRU4Rec, SASRec) could capture. These models learn sequential patterns that standard CF ignores, and applying them here would require treating each tourist trip as a session — feasible given the transaction timestamps.

OpenCLIP ViT-B-32 is a general-purpose model not optimized for retail product images. Fine-tuning on labeled product image-query pairs from this domain might improve intra-category discrimination and reduce false positives at category boundaries, if such training data could be assembled from the catalog.

A user study with real travelers would provide the turn-level metrics (task success, dialogue efficiency, satisfaction) needed to rigorously evaluate the conversational component. A lab study with participants from the platform's user base is the most feasible approach.

Finally, the demographic models should be audited for differential recommendation quality across nationality groups. If the model consistently recommends higher-quality stores to some national groups than others, this is an algorithmic fairness failure that should be quantified and reported before any production deployment.

---

## Chapter 8: Conclusion

### 8.1 Summary of Contributions

This thesis designed, implemented, and evaluated a conversational travel assistant for personalized store and product recommendations within a tax-free shopping ecosystem in northern Italy. The system addresses the specific constraints of the tourist shopping context: cold-start users with minimal purchase history, an interaction matrix that is 99.81% sparse, a multilingual cross-merchant product catalog requiring unified search, and the need for a natural language interface that travelers can use without prior familiarity with the system.

Four main contributions result from this work.

**First**, a working end-to-end web application that integrates OpenCLIP product search, 13 store recommendation models, and a GPT-4o-mini conversational interface into a deployed system accessible through a React frontend. The implementation demonstrates that the full pipeline — from raw transaction data through to real-time recommendations — is technically feasible within the constraints of the platform's data infrastructure.

**Second**, a systematic evaluation of 13 recommendation models across two levels (product and store) using seven metrics, with a rigorously leak-free temporal split protocol. The evaluation methodology corrects the data leakage flaw that leave-one-out cross-validation introduces for model-based CF, and the category-based product relevance judgment provides a more meaningful accuracy estimate than same-merchant relevance would.

**Third**, an empirical analysis of store-level CF under extreme sparsity, demonstrating that LightFM with WARP loss (HR@10 = 0.248) outperforms memory-based CF and all baselines in this setting. The results provide specific, quantified guidance: model-based approaches are necessary when average user history is below two interactions, and WARP's ranking-focused objective outperforms ALS's reconstruction-focused objective for top-k recommendation tasks.

**Fourth**, an empirical analysis of demographic feature augmentation, demonstrating that nationality and tourist type provide cold-start recommendation capability (HR@10 = 0.212 for LightFM Demo) competitive with behavioral CF for users who have interaction history, while providing uniquely useful signal for users with no history at all.

### 8.2 Answering the Research Questions

RQ1: Can multimodal CLIP embeddings enable effective cross-merchant product recommendations in a sparse, multilingual retail catalog?

Yes. OpenCLIP ViT-B-32 achieves HR@10 = 0.956 and nDCG@10 = 0.693 under a category-based self-retrieval evaluation protocol, well above both a random baseline (HR@10 = 0.435) and a popularity baseline (HR@10 = 0.456). The model achieves high catalog coverage (98.1%) and moderate novelty (5.662), demonstrating that it distributes recommendations across the full catalog rather than concentrating on a popular subset. These results confirm that shared image-text embeddings generalize effectively across merchant boundaries and multilingual content without any domain-specific fine-tuning.

RQ2: Can store-level CF produce meaningful personalized recommendations despite extreme sparsity?

Partially. LightFM WARP achieves HR@10 = 0.248, a thirteen-fold improvement over the random baseline (0.019) and a 60% improvement over the global popularity baseline (0.155). Model-based approaches clearly outperform memory-based CF under the dataset's sparsity conditions: cosine similarity on near-empty interaction vectors is unreliable, making ALS and LightFM the appropriate model families. However, the absolute hit rate — one in four test users has a future store visit correctly predicted in the top ten — reflects the fundamental difficulty of predicting behavior from 1.42 average store visits. Meaningful signal is present, but the recommendation task is hard by construction.

RQ3: Do demographic features improve store recommendations over behavioral approaches?

Not substantially for users with behavioral history. LightFM Demo (demographic-only) achieves HR@10 = 0.212, nearly identical to LightFM Hybrid (behavioral, HR@10 = 0.214). LightFM Full Hybrid (HR@10 = 0.230) is below pure LightFM WARP (HR@10 = 0.248). Demographic features provide cold-start capability — Demographic Popularity (HR@10 = 0.174) outperforms the popularity baseline (HR@10 = 0.155) without any behavioral data — but do not improve accuracy for users who already have interaction records.

RQ4: Can a tool-calling LLM effectively mediate between user preferences and a CLIP search index?

Functionally, yes. The tool-calling architecture enables structured preference elicitation, dynamic query formulation, and intelligent result curation within a natural language interface. The select_products mechanism ensures that recommendation carousels contain contextually appropriate items rather than fixed-length nearest-neighbor lists. Formal quantitative evaluation would require user studies, but the qualitative behavior across a range of query types demonstrates that the integration is viable and addresses limitations inherent in pure retrieval systems.

### 8.3 Closing Remarks

The tax-free ecosystem is an unusual setting for recommendation research: the cold-start problem is not a special case but the norm, the interaction matrix is sparser than almost any retail dataset studied in the CF literature, and the goal of the recommendation must be actionable within a short visit window. These constraints simultaneously make the problem harder and make the solutions more consequential. A system that successfully predicts which new stores a traveler will enjoy — correctly in one of four cases, as LightFM WARP achieves — is doing something genuinely useful for a person navigating an unfamiliar city with limited time.

The content-based layer powered by CLIP embeddings is the most reliable component of the current system and the one best aligned with the cold-start reality. Its accuracy is high, its catalog coverage is broad, and it requires no interaction history. Extending catalog coverage to additional merchants within the tax-free network represents the highest-value near-term improvement: the infrastructure is ready, and the benefit would scale directly with the number of merchants added.

The broader convergence of large-scale pretrained vision-language models, efficient implicit-feedback matrix factorization, and instruction-following language models makes this kind of integrated recommendation assistant feasible at modest cost. Deploying a system of this kind in an operational tax-free ecosystem would both improve the traveler experience and generate the interaction data — real clicks, real purchases, real refinements — needed to improve it further.

---

## References

1. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. *IEEE Transactions on Knowledge and Data Engineering*, 17(6), 734–749.

2. Adomavicius, G., Mobasher, B., Ricci, F., & Tuzhilin, A. (2011). Context-aware recommender systems. *AI Magazine*, 32(3), 67–80.

3. Adomavicius, G., & Kwon, Y. (2012). Improving aggregate recommendation diversity using ranking-based techniques. *IEEE Transactions on Knowledge and Data Engineering*, 24(5), 896–911.

4. Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.

5. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., ... & Jitsev, J. (2023). Reproducible scaling laws for contrastive language-image learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)*.

6. Deshpande, M., & Karypis, G. (2004). Item-based top-N recommendation algorithms. *ACM Transactions on Information Systems*, 22(1), 143–177.

7. European Commission. (n.d.). VAT refunds for tourists. Retrieved from the European Commission official website.

8. Gao, C., Lei, W., He, X., de Rijke, M., & Chua, T.-S. (2021). Advances and challenges in conversational recommender systems: A survey. *AI Open*, 2, 100–126.

9. Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. *ACM Transactions on Information Systems*, 22(1), 5–53.

10. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM 2008)*, 263–272.

11. Hwangbo, H., Kim, Y. S., & Cha, K. J. (2018). Recommendation system development for fashion retail e-commerce. *Electronic Commerce Research and Applications*, 28, 94–101.

12. Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. *Egyptian Informatics Journal*, 16(3), 261–273.

13. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422–446.

14. Jannach, D., Manzoor, A., Cai, W., & Chen, L. (2021). A survey on conversational recommender systems. *ACM Computing Surveys*, 54(5), 1–36.

15. Javed, U., Shaukat, K., Hameed, I. A., Iqbal, F., Alam, T. M., & Luo, S. (2021). A review of content-based and context-based recommendation systems. *International Journal of Emerging Technologies in Learning*, 16(3), 274–306.

16. Kula, M. (2015). Metadata embeddings for user and item cold-start recommendations. *Proceedings of the 2nd Workshop on New Trends on Content-Based Recommender Systems (CBRecSys) at RecSys 2015*.

17. Lika, B., Kolomvatsos, K., & Hadjiefthymiades, S. (2014). Facing the cold start problem in recommender systems. *Expert Systems with Applications*, 41(4), 2065–2073.

18. Luo, S., Chen, M., Li, K., & Yin, J. (2020). Deep critiquing for VAE-based recommender systems. *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2020)*.

19. Maffini, F., & Ricci, V. (2019). Tax-free consumption in the EU. Global Blue / Altagamma.

20. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

21. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*.

22. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. *Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009)*.

23. Schafer, J. B., Konstan, J. A., & Riedl, J. (2001). E-commerce recommendation applications. *Data Mining and Knowledge Discovery*, 5(1), 115–153.

24. Son, L. H. (2016). Dealing with the new user cold-start problem in recommender systems: A comparative review. *Information Systems*, 58, 87–104.

25. Stalidis, G., Karapapas, C., Diamantaras, K., Papadopoulos, S., & Tzovaras, D. (2023). Recommendation systems for e-shopping: Review of techniques for retail and sustainable marketing. *Sustainability*, 15(23), 16151.

26. Sun, Y., & Zhang, Y. (2018). Conversational recommender system. *Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2018)*, 235–244.

27. Vargas, S., & Castells, P. (2011). Rank and relevance in novelty and diversity metrics for recommender systems. *Proceedings of the 5th ACM Conference on Recommender Systems (RecSys 2011)*, 109–116.

28. Ziegler, C.-N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). Improving recommendation lists through topic diversification. *Proceedings of the 14th International World Wide Web Conference (WWW 2005)*, 22–32.
