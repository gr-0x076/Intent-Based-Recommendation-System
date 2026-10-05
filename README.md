# Intent-Based E-Commerce Recommender System

> **Guiding Principle:** *The goal is not to recommend what users liked in the past. The goal is to recommend what users need right now.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Phase 2 Complete](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen.svg)]()

---

## 1. Project Overview & Motivation

In e-commerce, user journeys are dynamic, fast-paced, and frequently anonymous. Traditional recommender systems depend heavily on long-term user profiles and static collaborative filtering. In practice, this creates major failure modes:
1. **Historical Bias:** Recommending items a user already bought or was interested in weeks ago.
2. **Context Blindness:** Inability to recognize that a user's goals change between morning work browsing and evening leisure shopping.
3. **Severe Cold-Start:** When a user visits without an account or history, traditional systems fall back to generic popularity.

Having studied classical recommendation techniques (collaborative filtering, matrix factorization) and sequential deep learning architectures (GRU4Rec, SASRec), this project designs an **Intent-Based Recommender System**. Rather than asking *"what did this user like in the past?"*, our system asks:

> **"What is the user trying to achieve in this specific session right now?"**

We formulate this as an **evidence-driven research investigation**, advancing through controlled experimental baselines before introducing multi-modal intent representations.

---

## 2. How the Recommendation System Works

### 2.1 What is a Recommender System?
At its core, a recommender system is a ranking engine: given a catalog of tens of thousands of products and a stream of user actions (page views, clicks, searches, cart additions), it predicts the most relevant items the user will interact with next.

### 2.2 Our Two-Stage Pipeline Architecture

```text
                  Current Session Actions
      [Search: "running shoes"] → [View: Shoe A] → [View: Shoe B]
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        Stage 1: Multi-Recall            Stage 1: Intent Retrieval
       (ItemCF Co-occurrence)          (Semantic Query / Content Vector)
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                   Merged Candidate Pool (~100 items)
                                 │
                 Stage 2: Intent-Aware Re-ranking
             (Recency Decay + Action Escalation + Query Match)
                                 │
                    Top-20 Recommendations
```

* **Stage 1 (Fast Candidate Retrieval):** Scans the large product catalog ($66,000+$ items) in milliseconds to retrieve the top 100 candidate items using complementary retrieval strategies:
  * *Session ItemCF:* Pulls items frequently viewed alongside recent session products.
  * *Dense Intent Retrieval:* Pulls items semantically aligned with the user's active search query vector.
* **Stage 2 (Intent-Aware Re-ranking):** Scores and re-ranks the candidates by evaluating behavioral recency, transition velocity, and stage of commitment (e.g., browsing vs. comparing vs. cart addition).

---

## 3. Dataset & Empirical Discoveries

We evaluate our research on a large-scale e-commerce dataset containing over **36 million user events** across ~5 million sessions:
* `browsing_train.csv`: 36,079,308 browsing logs (pageviews, product detail views, cart additions, purchases).
* `search_train.csv`: 819,517 on-site search events, each enriched with a 50-dimensional dense pre-trained query vector.
* `sku_to_content.csv`: 66,386 catalog products with 50-dimensional product content vectors.

### Critical Empirical Profiling Insights

Before writing any model code, we conducted data profiling across 2 million raw interactions to understand the actual session dynamics:

1. **Cold-Start is the Dominant Reality:**
   * **0 product interactions:** 33.37% (91,671 sessions) — *Pure navigation / bounces.*
   * **1 product interaction:** 34.17% (93,860 sessions) — *Immediate departure after single view.*
   * **2 product interactions:** 10.30% (28,307 sessions) — *Single input, single target.*
   * **3+ product interactions:** 22.16% (60,886 sessions) — *Multi-step session.*
   * **Takeaway:** **77.84% of sessions have $\le 2$ product interactions.** The recommender must excel in ultra-short, sparse contexts.

2. **Severe Action Imbalance:**
   * `pageview` (no product SKU): **70.6%**
   * `detail` (product view): **27.5%**
   * `add` to cart: **0.9%** (present in only 4.3% of sessions)
   * `purchase`: **0.2%** (present in only 1.1% of sessions)
   * **Takeaway:** The prediction target must be formulated as the *next product interaction*, while pageviews provide navigation context.

3. **Query Vector Characteristics:**
   * Query vectors and product vectors share a 50-dimensional space, but empirical cosine similarity on clicked products is centered around zero (mean $\approx -0.0285$). Uncalibrated raw dot products do not work as zero-shot rankers; learned intent projection is required.

---

## 4. Progress So Far: Official Baseline Ladder

To rigorously test whether intent modeling genuinely adds value, we established a strict baseline ladder. All models are trained on 105,109 sessions and evaluated on the exact same **24,968 prefix-expanded validation examples** (chronological split) using our standardized evaluation harness (`src/evaluation/metrics.py`).

```
=======================================================================================================================
                                             OFFICIAL BENCHMARK LADDER (N = 24,968)
=======================================================================================================================
Model                         MRR@20    Recall@20   Cold-1 MRR   Cold-2 MRR   Rich 3+ MRR   Has-Search   No-Search
-----------------------------------------------------------------------------------------------------------------------
Level 0: Global Popularity    0.0017     0.0074       0.0008       0.0024       0.0019        0.0033       0.0015
Level 1: ItemCF (Decay=0.7)   0.1440     0.2767       0.1875       0.1541       0.1171        0.1100       0.1480
Level 2: GRU4Rec (PyTorch)    0.1161     0.2011       0.1525       0.1249       0.0934        0.0835       0.1199
=======================================================================================================================
```

### Research Findings & Diagnostic Revelations

1. **Why ItemCF Outperforms Pure GRU4Rec in This Domain:**
   * In sparse, short sessions ($77.8\%$ have $\le 2$ product interactions), direct statistical co-occurrence with recency decay ($0.7^j$) is a stronger inductive prior than neural sequence transitions learned from scratch on 10,630 items.
   * A single-layer GRU struggles to generalize on sequences of length 1 or 2 items, confirming our hypothesis that complex neural sequence models require hybridization with collaborative heuristics.

2. **The "Search Intent Deficit":**
   * **ItemCF:** Drops from **0.1480** (No-search) to **0.1100** (Has-search) — a **25.7% relative drop**.
   * **GRU4Rec:** Drops from **0.1199** (No-search) to **0.0835** (Has-search) — a **30.4% relative drop**.
   * *Diagnosis:* When a user issues a search query, their goal shifts. Both baselines are blind to the search query vector and continue recommending based solely on past item views.

3. **The "Action Escalation Blindspot":**
   * While ItemCF achieves $0.1530$ MRR on target `detail` views, it drops to **$0.0065$ on `add-to-cart`** and **$0.0181$ on `purchase`**.
   * Transitioning from casual browsing to cart addition represents an intent escalation that neither co-occurrence nor pure sequence modeling captures.

---

## 5. Codebase Structure

```
intent-rec/
├── data/                     # Data directory (ignored from Git)
│   ├── raw/                  # Symlinks to raw dataset CSVs
│   └── processed/            # Sampled sessions & validation examples (pkl/parquet)
├── docs/                     # Research design & architectural documents
│   ├── recommendation_paradigms.md       # Comparative survey of recsys paradigms
│   ├── Intent_Based_Recommender_Clarity.md # Conceptual & theoretical foundations
│   ├── Project_Plan.md                   # Full phased research plan
│   └── Updated_Plan_After_EDA.md         # Empirical data profiling report
├── src/
│   ├── data/
│   │   ├── pipeline.py       # Base data pipeline interface
│   │   └── sampler.py        # Chunk-based streaming sampler with deterministic hash
│   ├── models/
│   │   ├── popularity.py     # Level 0: Global popularity with seen-item filter
│   │   ├── itemcf.py         # Level 1: Normalized ItemCF with 0.7^j recency decay
│   │   ├── gru4rec.py        # Level 2: Canonical PyTorch GRU sequence model
│   │   ├── session_encoder.py# Neural session encoder module
│   │   └── train.py          # Training loop utilities
│   ├── retrieval/
│   │   └── faiss_index.py    # Fast approximate nearest neighbor index
│   ├── api/
│   │   └── main.py           # FastAPI real-time serving endpoints
│   └── evaluation/
│       ├── metrics.py        # Standardized evaluation harness (MRR@K, Recall@K, segments)
│       └── run_benchmarks.py # Automated multi-model benchmark runner
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis notebook
│   └── findings.md           # Dataset profiling statistics
├── results/
│   └── baselines.md          # Official benchmark comparison table & segment metrics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. How to Reproduce

### 1. Installation

```bash
git clone https://github.com/gr-0x076/Intent-Based-Recommendation-System.git
cd Intent-Based-Recommendation-System
pip install -r requirements.txt
```

### 2. Stream & Sample the Dataset

Link your raw CSV files to `data/raw/` and run the streaming sampler (<300MB RAM usage):

```bash
python src/data/sampler.py --sample-pct 3.0 --output-dir data/processed/dev_3pct
```

### 3. Run Benchmark Ladder

Evaluate all baseline models on the 24,968 validation examples:

```bash
python src/evaluation/run_benchmarks.py --data-dir data/processed/dev_3pct
```

---

## 7. Current Project Status & Next Steps

```
[Phase 0] Data Engineering & Evaluation Harness          ✅ COMPLETE
[Phase 1] Level 0 (Popularity) & Level 1 (ItemCF)        ✅ COMPLETE (MRR: 0.1440)
[Phase 2] Level 2 (GRU4Rec) Sequential Baseline          ✅ COMPLETE (MRR: 0.1161)
[Phase 3] Isolated Intent Experiments                    🚀 CURRENT FOCUS
    ├── 3A: Behavioral Action Progression Signals
    ├── 3B: Search Query Vector Alignment (Targeting the 0.1100 search deficit)
    └── 3C: Hybrid Intent Fusion
[Phase 4] Dense Retrieval (FAISS) & Re-ranking (LightGBM) ⏳ PLANNED
[Phase 5] Real-Time Interactive Demo (FastAPI + Streamlit)⏳ PLANNED
```
