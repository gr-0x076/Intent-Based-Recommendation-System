# Intent-Based E-Commerce Recommender System

> **Guiding Principle:** *The goal is not to recommend what users liked in the past. The goal is to recommend what users need right now.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Phase 3 In Progress](https://img.shields.io/badge/Status-Phase%203%20In%20Progress-blue.svg)]()

---

## 1. Project Overview & Research Question

Traditional recommender systems rely predominantly on long-term historical user profiles and static collaborative filtering. In modern e-commerce journeys, however, users browse anonymously, exhibit rapid goal shifts, and leave short interaction trails where historical profiles are unavailable or uninformative.

This project investigates:
> **Does explicitly modeling current-session user intent (via query vectors, interaction transitions, and commitment states) produce measurably better recommendations than sequence ordering and collaborative filtering alone?**

We follow a rigorous, **evidence-driven research methodology**:
```
Level 0: Global Popularity (Floor Baseline)
   ↓
Level 1: ItemCF (Session Co-occurrence + Exponential Recency Decay)
   ↓
Level 2: Pure GRU4Rec (Sequential RNN Baseline)
   ↓
Level 3A: GRU + Raw Features (Control: Concatenated Action + Query Vectors)
   ↓
Level 3B: GRU + Latent Intent Bottleneck (Treatment: Learned Intent Abstraction)
   ↓
Level 4: Dense Candidate Retrieval & Re-ranking (Only if empirically justified)
```

---

## 2. What is "Intent" in This System?

To avoid treating "intent" as an ambiguous buzzword, this project operationalizes intent into **two distinct behavioral dimensions**:

1. **Semantic / Goal Intent (Expressed via Queries & Navigation):**
   * What specific task or attribute is the user pursuing? (e.g., searching for *"waterproof hiking shoes"* vs. browsing casual footwear).
   * *Signal:* 50-dimensional search query vectors and contextual URL navigation paths.

2. **Commitment / Behavioral Stage (Expressed via Action Dynamics):**
   * Is the user in **Discovery Mode** (exploring novel items across categories) or **Conversion Mode** (re-evaluating an examined product for cart addition or checkout)?
   * *Signal:* Transition patterns between `pageview`, `detail`, `add-to-cart`, and repeat interaction velocity.

---

## 3. Current Architecture Hypothesis

The architecture below represents our **working research hypothesis**, not an established conclusion. Components will be retained, adapted, or discarded based strictly on empirical ablation evidence.

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
                           (Recency Decay + Stage Adaptation + Query Match)
                                               │
                                  Top-20 Recommendations
```

* **Candidate Retrieval (Hypothesis):** Combining co-occurrence heuristics with semantic query retrieval will expand candidate recall, especially when active search signals an intent shift away from prior browsing.
* **Re-ranking (Hypothesis):** Scoring candidates by explicitly distinguishing between discovery items (novel recommendations) and commitment items (repeat consideration) will bridge the gap between browsing and conversion.

---

## 4. Empirical Dataset Discoveries

We conduct our research on a large-scale e-commerce dataset containing over **36 million user events** across ~5 million sessions:
* `browsing_train.csv`: 36,079,308 browsing logs (pageviews, product detail views, cart additions, purchases).
* `search_train.csv`: 819,517 on-site search events, each with a 50-dimensional dense pre-trained query vector.
* `sku_to_content.csv`: 66,386 catalog products with 50-dimensional product content vectors.

### Empirical Profiling (from 2M raw rows sample)

1. **Cold-Start is the Dominant Mode:**
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
   * **Takeaway:** The prediction target must be formulated as the *next product interaction*, preserving action type `(target_item, target_action)`.

3. **Query Vector Characteristics:**
   * Query vectors and product vectors share a 50-dimensional space, but direct cosine similarity on clicked products is centered around zero (mean $\approx -0.0285$). Initial analysis suggests that raw query-product similarity is not sufficiently aligned for zero-shot retrieval; learned alignment will be tested.

---

## 5. Official Benchmark Ladder & Empirical Findings

All models are trained on 105,109 sessions and evaluated on the exact same **24,968 prefix-expanded validation examples** (chronological split) using our standardized evaluation harness (`src/evaluation/metrics.py`).

```
========================================================================================================================================
                                             OFFICIAL BENCHMARK LADDER (N = 24,968)
========================================================================================================================================
Model                         MRR@20    Recall@20   Cold-1 MRR   Cold-2 MRR   Rich 3+ MRR   Has-Search   No-Search   Discovery   Conversion
----------------------------------------------------------------------------------------------------------------------------------------
Level 0: Global Popularity    0.0017     0.0074       0.0008       0.0024       0.0019        0.0033       0.0015     0.0028      0.0000
Level 1: ItemCF (Decay=0.7)   0.1440     0.2767       0.1875       0.1541       0.1171        0.1100       0.1480     0.2489      0.0000
Level 2: Pure GRU4Rec (RNN)   0.1161     0.2011       0.1525       0.1249       0.0934        0.0835       0.1199     0.1944      0.0000
Level 3A: GRU + Raw Features  0.1216     0.2130       0.1609       0.1300       0.0974        0.0896       0.1253     0.2035      0.0000
[Sanity] Repeat Last Item     0.1896     0.1896       0.2634       0.1871       0.1510        0.1923       0.1893     0.0000      0.4708
========================================================================================================================================
```

### Critical Scientific Insights

1. **Model 3A (Control) Performance:**
   * Adding raw action embeddings (16-D) and search query projections (32-D) via standard concatenation improves GRU performance from **0.1161 to 0.1216 MRR@20** ($+4.7\%$ relative gain).
   * **The Phase 3 Test:** Model 3A establishes the essential control baseline. When Model 3B (Latent Intent Bottleneck) is evaluated, the hypothesis test is **3B vs. 3A (0.1216)**. Only gains exceeding $0.1216$ can be scientifically attributed to intent abstraction rather than simple feature augmentation.

2. **Why ItemCF Outperforms GRU4Rec Under Our Current Setup:**
   * In sparse, short sessions ($77.8\%$ have $\le 2$ product interactions), direct statistical co-occurrence with recency decay ($0.7^j$) is a stronger inductive prior than neural sequence transitions learned from scratch on 10,630 items.
   * Our current single-layer GRU configurations (both pure sequence at $0.1161$ and feature-augmented at $0.1216$) trail ItemCF on early cold-start transitions ($0.1609$ vs $0.1875$).

3. **Stratified Search-Session Analysis (Controlling for Confounders):**
   * **Unstratified Gap:** ItemCF drops from **0.1480** (No-search) to **0.1100** (Has-search).
   * **Confounder Isolated:** 97.8% of searches occur in long sessions (`len_6_plus`).
   * **Controlled Delta:** ItemCF performance remains lower on search sessions even after controlling for session length, but the effect is much smaller than the raw comparison suggests:
     $$\Delta \text{ MRR@20} = -0.0138, \quad 95\% \text{ bootstrap CI } [-0.0256, -0.0024]$$

4. **Action-Target Gap & The Repeat-Item Discovery:**
   * Under standard discovery evaluation with seen-item filtering (`filter_seen=True`), ItemCF scored **0.0065 on add-to-cart** and **0.0181 on purchase**.
   * *Sanity Check Investigation:* Analyzing target distribution revealed that **95.0% of add-to-cart targets** and **94.3% of purchase targets** are items **already seen in the session prefix** (92.2% and 78.2% being the *exact last item viewed*).
   * A trivial `Repeat Last Item` baseline achieves **0.9219 MRR on add-to-cart** and **0.7824 MRR on purchase**!
   * *Dual-Mode Requirement:* Recommender systems require dual-mode intent modeling:
     * **Discovery Intent (New Items):** Predicts novel items to explore ($59.7\%$ of cases, ItemCF MRR = 0.2489).
     * **Conversion Intent (Repeat Items):** Predicts when the user transitions from exploring to purchasing an examined item ($40.3\%$ of cases, Repeat Prior MRR = 0.4708).

---

## 6. Codebase Structure

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
│   │   ├── gru4rec_features.py# Level 3A: GRU + Raw Features (Phase 3 Control)
│   │   ├── session_encoder.py# Neural session encoder module
│   │   └── train.py          # Training loop utilities
│   ├── retrieval/
│   │   └── faiss_index.py    # Fast approximate nearest neighbor index
│   ├── api/
│   │   └── main.py           # FastAPI real-time serving endpoints
│   └── evaluation/
│       ├── metrics.py        # Standardized evaluation harness (MRR@K, Recall@K, dual-mode segments)
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

## 7. How to Reproduce

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

## 8. Current Project Status & Next Steps

```
[Phase 0] Data Engineering & Evaluation Harness          ✅ COMPLETE
[Phase 1] Level 0 (Popularity) & Level 1 (ItemCF)        ✅ COMPLETE (MRR: 0.1440)
[Phase 2] Level 2 (GRU4Rec) Sequential Baseline          ✅ COMPLETE (MRR: 0.1161)
[Phase 3] Isolated Intent Experiments                    🚀 CURRENT FOCUS
    ├── 3A: GRU + Raw Features Control                   ✅ COMPLETE (MRR: 0.1216)
    ├── 3B: GRU + Latent Intent Bottleneck               ⏳ NEXT (The 3B vs 3A Test)
    └── 3C: Dual-Mode Intent Fusion (Discovery vs Conversion)
[Phase 4] Dense Retrieval & Re-ranking (FAISS / LightGBM)⏳ PLANNED (Only if justified)
[Phase 5] Real-Time Interactive Demo (FastAPI + Streamlit)⏳ PLANNED
```
