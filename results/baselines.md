# Baseline Results

Evaluated on the exact same chronological validation set ($N = 24,968$ prefix-expanded session examples from 22,524 validation sessions).

## Official Baseline Ladder

| Level | Model | MRR@20 | Recall@20 | Cold-Start-1 | Cold-Start-2 | Rich 3+ | Has-Search | No-Search |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Level 0 | Global Popularity (Filter Seen=True) | 0.0017 | 0.0074 | 0.0008 | 0.0024 | 0.0019 | 0.0033 | 0.0015 |
| Level 1 | ItemCF (Co-occurrence + Recency Decay $0.7^j$) | 0.1440 | 0.2767 | 0.1875 | 0.1541 | 0.1171 | 0.1100 | 0.1480 |
| Level 2 | GRU4Rec (PyTorch Canonical RNN) | 0.1161 | 0.2011 | 0.1525 | 0.1249 | 0.0934 | 0.0835 | 0.1199 |
| [Sanity] | Repeat Last Item (Recency Prior) | 0.1896 | 0.1896 | 0.2634 | 0.1871 | 0.1510 | 0.1923 | 0.1893 |
| Level 3 | Intent Model (Query + Action Experiments) | *Pending Phase 3* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

## Detailed Segment Breakdown

### Level 1: ItemCF ($N = 24,968$)
* **Cold-Start ($n=1$ prior product):** MRR@20 = 0.1875, Recall@20 = 0.3421 (Count: 7,059)
* **Cold-Start ($n=2$ prior products):** MRR@20 = 0.1541, Recall@20 = 0.2962 (Count: 4,767)
* **Rich ($n \ge 3$ prior products):** MRR@20 = 0.1171, Recall@20 = 0.2344 (Count: 13,142)
* **Has Search:** MRR@20 = 0.1100, Recall@20 = 0.2317 (Count: 2,611)
* **No Search:** MRR@20 = 0.1480, Recall@20 = 0.2819 (Count: 22,357)
* **Target Detail View:** MRR@20 = 0.1530, Recall@20 = 0.2936 (Count: 23,368)
* **Target Add-to-Cart:** MRR@20 = 0.0065, Recall@20 = 0.0138 (Count: 1,088)
* **Target Purchase:** MRR@20 = 0.0181, Recall@20 = 0.0259 (Count: 193)

### Level 2: GRU4Rec ($N = 24,968$)
* **Cold-Start ($n=1$ prior product):** MRR@20 = 0.1525, Recall@20 = 0.2607
* **Cold-Start ($n=2$ prior products):** MRR@20 = 0.1249, Recall@20 = 0.2161
* **Rich ($n \ge 3$ prior products):** MRR@20 = 0.0934, Recall@20 = 0.1636
* **Has Search:** MRR@20 = 0.0835, Recall@20 = 0.1520
* **No Search:** MRR@20 = 0.1199, Recall@20 = 0.2068
* **Target Detail View:** MRR@20 = 0.1232, Recall@20 = 0.2132
* **Target Add-to-Cart:** MRR@20 = 0.0039, Recall@20 = 0.0083
* **Target Purchase:** MRR@20 = 0.0192, Recall@20 = 0.0259

### Sanity Check: Repeat Last Item Baseline ($N = 24,968$)
* **Overall MRR@20:** 0.1896
* **Target Add-to-Cart:** **MRR@20 = 0.9219** (Count: 1,088)
* **Target Purchase:** **MRR@20 = 0.7824** (Count: 193)
* **Target Detail View:** MRR@20 = 0.1482 (Count: 23,368)

## Core Observations & Hypotheses

1. **ItemCF outperforms GRU4Rec on short sessions:** In sparse sessions ($77.8\%$ have $\le 2$ product interactions), statistical co-occurrence with recency decay provides a more effective prior than a sequence RNN trained from scratch on 10.6k vocabulary items.
2. **Search-Session Performance Gap:** Both ItemCF and GRU4Rec show a severe drop on sessions containing search events ($0.1480 \to 0.1100$ and $0.1199 \to 0.0835$). We hypothesize that query information contains explicit intent constraints that current behavioral baselines fail to leverage.
3. **Action-Target Gap & The Repeat-Item Discovery:**
   - Under seen-item filtering (`filter_seen=True`), baselines collapse on add-to-cart ($0.0065$) and purchase ($0.0181$).
   - Empirical repeat-item analysis reveals that **95.0% of cart additions** and **94.3% of purchases** are items already seen in the session prefix.
   - A pure "Repeat Last Item" baseline achieves **0.9219 MRR on add-to-cart** and **0.7824 MRR on purchase**.
   - **Conclusion:** Recommender systems require dual-mode intent modeling: **Discovery Mode** (novel item recommendation) vs. **Conversion Mode** (re-engaging examined items).
