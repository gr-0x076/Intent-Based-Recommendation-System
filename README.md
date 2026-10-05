# Intent-Based E-Commerce Recommender System

> **Guiding Principle:** *The goal is not to recommend what users liked in the past. The goal is to recommend what users need right now.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Phase 2 Complete](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen.svg)]()

---

## 1. Project Overview & Research Question

Traditional recommender systems rely predominantly on long-term historical profiles and static collaborative filtering. In modern e-commerce, users exhibit dynamic, context-dependent journeys with frequent intent shifts and short sessions where past history is either absent (cold-start) or irrelevant.

This project investigates:
> **Does explicitly modeling current-session user intent (via query vectors, interaction types, and behavioral escalation) produce measurably better recommendations than sequence ordering and collaborative filtering alone?**

We follow a rigorous, **evidence-driven research methodology**:
```
Level 0: Global Popularity (Floor Baseline)
   ↓
Level 1: ItemCF (Session Co-occurrence + Exponential Recency Decay)
   ↓
Level 2: GRU4Rec (Canonical Sequential RNN Baseline)
   ↓
Level 3: Isolated Intent Experiments (Action Signals, Query Vectors, Hybrid Fusion)
   ↓
Level 4: Dense Candidate Retrieval & Re-ranking (FAISS + LightGBM)
```

---

## 2. Dataset & Empirical Profiling

We use the [Coveo SIGIR eCom 2021 Data Challenge](https://github.com/coveooss/SIGIR-ecom-data-challenge) dataset containing real-world e-commerce user journeys:

* **Browsing Logs (`browsing_train.csv`):** 36,079,308 rows (~5M sessions, 6.4 GB).
* **Search Logs (`search_train.csv`):** 819,517 query events with 50-D pre-trained dense query vectors (1.7 GB).
* **Product Catalog (`sku_to_content.csv`):** 66,386 products with 50-D content vectors and metadata.

### Empirical Data Discoveries (from 2M raw rows sample)

1. **The Cold-Start Dominance:**
   * **0 product interactions:** 33.37% (91,671 sessions) — *Pure navigation / bounces.*
   * **1 product interaction:** 34.17% (93,860 sessions) — *Immediate exit.*
   * **2 product interactions:** 10.30% (28,307 sessions) — *Single input, single target.*
   * **3+ product interactions:** 22.16% (60,886 sessions) — *Rich trajectory.*
   * **$\le 2$ product interactions:** **77.84%** of all sessions! Cold-start is not an edge case; it is the dominant mode.

2. **Action Sparsity:**
   * `pageview` (no SKU): **70.6%**
   * `detail` (product view): **27.5%**
   * `add-to-cart`: **0.9%** (present in only 4.3% of sessions)
   * `purchase`: **0.2%** (present in only 1.1% of sessions)

3. **Query Vector Compatibility:**
   * Query vectors and product description vectors are both 50-dimensional (`float32`), but empirical cosine similarity on clicked products is centered around zero (mean $\approx -0.0285$). They cannot be naively scored via raw uncalibrated dot products without learned alignment.

---

## 3. Official Benchmark Ladder

All models are evaluated on the exact same **chronological validation split** (105,109 train sessions, 22,524 validation sessions) evaluated on **24,968 prefix evaluation examples** using our standardized evaluation harness (`src/evaluation/metrics.py`).

```
=======================================================================================================================
                                             OFFICIAL BENCHMARK LADDER (N = 24,968)
=======================================================================================================================
Model                         MRR@20    Recall@20   Cold-1 MRR   Cold-2 MRR   Rich 3+ MRR   Has-Search   No-Search
-----------------------------------------------------------------------------------------------------------------------
Level 0: Global Popularity    0.0017     0.0074       0.0008       0.0024       0.0019        0.0033       0.0015
Level 1: ItemCF (Decay=0.7)   0.1440     0.2767       0.1875       0.1541       0.1171        0.1100       0.1480
Level 2: GRU4Rec (RNN)        0.1161     0.2011       0.1525       0.1249       0.0934        0.0835       0.1199
=======================================================================================================================
```

### Key Research Findings

1. **Why ItemCF Outperforms Pure GRU4Rec in This Setting:**
   * In sparse, short sessions ($77.8\%$ have $\le 2$ product events), direct statistical co-occurrence with recency decay ($0.7^j$) serves as a stronger inductive prior than neural sequence transitions learned from scratch on 10,630 items.
   * A single-layer GRU struggles to generalize on sequence lengths of 1 or 2 items, explaining why competitive solutions (e.g. DeepBlueAI 1st place SIGIR 2021) prioritized multi-recall ItemCF over complex RNNs.

2. **The "Search Intent Deficit":**
   * **ItemCF:** Drops from **0.1480** (No-search) to **0.1100** (Has-search) — a **25.7% relative drop**.
   * **GRU4Rec:** Drops from **0.1199** (No-search) to **0.0835** (Has-search) — a **30.4% relative drop**.
   * *Observation:* Both baseline models exhibit severe degradation when search events occur, suggesting that search-related intent is poorly captured by sequence order and item co-occurrence alone.

3. **The "Action Escalation Blindspot":**
   * While ItemCF achieves $0.1530$ MRR on target `detail` views, it drops to **$0.0065$ on `add-to-cart`** and **$0.0181$ on `purchase`**.
   * Interaction type and behavioral stage represent critical intent signals that standard baselines fail to model.

---

## 4. Codebase Architecture

```
intent-rec/
├── data/
│   ├── raw/                  # Symlinks to original Coveo dataset CSVs
│   └── processed/            # Sampled sessions & validation examples (pkl/parquet)
├── docs/                     # Research analysis & architectural documentation
│   ├── SIGIR_eCOM_2021_Report.md         # 1st-place solution deep dive
│   ├── Intent_Based_Recommender_Clarity.md # Conceptual & theoretical foundations
│   ├── Project_Plan.md                   # Initial comprehensive project plan
│   └── Updated_Plan_After_EDA.md         # Data-driven revised plan after profiling
├── src/
│   ├── data/
│   │   └── sampler.py        # Chunk-based streaming sampler with deterministic hash
│   ├── models/
│   │   ├── popularity.py     # Level 0 baseline: global frequency with seen filter
│   │   ├── itemcf.py         # Level 1 baseline: normalized co-occurrence + 0.7^j decay
│   │   └── gru4rec.py        # Level 2 baseline: canonical PyTorch GRU sequence model
│   └── evaluation/
│       ├── metrics.py        # Evaluation harness: MRR@K, Recall@K, segmented analysis
│       └── run_benchmarks.py # Benchmark runner comparing all models side-by-side
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. How to Reproduce

### 1. Installation

```bash
git clone https://github.com/gr-0x076/Intent-Based-Recommendation-System.git
cd Intent-Based-Recommendation-System
pip install -r requirements.txt
```

### 2. Stream & Sample Dataset

Link your raw dataset files into `data/raw/` and run the streaming sampler:

```bash
# Samples 3% of all sessions across 36M rows with zero RAM overflow (<300MB RAM)
python src/data/sampler.py --sample-pct 3.0 --output-dir data/processed/dev_3pct
```

### 3. Run Benchmark Ladder

Evaluate all baseline models across all 24,968 validation examples:

```bash
python src/evaluation/run_benchmarks.py --data-dir data/processed/dev_3pct
```

---

## 6. Current Roadmap

- [x] **Phase 0:** Data Profiling, Session Definition, Evaluation Harness
- [x] **Phase 1:** Level 0 (Popularity) & Level 1 (ItemCF) Baselines
- [x] **Phase 2:** Level 2 (GRU4Rec) Sequential Baseline
- [ ] **Phase 3:** Isolated Intent Modeling Experiments:
  - [ ] **3A:** Action-type behavioral progression (weighting & transition dynamics)
  - [ ] **3B:** Search query vector integration (closing the search intent deficit)
  - [ ] **3C:** Hybrid Intent Fusion
- [ ] **Phase 4:** Candidate Retrieval (FAISS) & Feature-Based Ranking (LightGBM)
- [ ] **Phase 5:** Live Demonstration & Real-time Session Simulator (FastAPI + Streamlit)
- [ ] **Phase 6:** Intent Shift Analysis & Technical Documentation
