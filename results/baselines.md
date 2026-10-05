# Baseline Results

Evaluated on the exact same chronological validation set ($N = 24,968$ prefix-expanded session examples from 22,524 validation sessions).

## Official Baseline Ladder

| Level | Model | MRR@20 | Recall@20 | Cold-Start-1 | Cold-Start-2 | Rich 3+ | Has-Search | No-Search |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Level 0 | Global Popularity (Filter Seen=True) | 0.0017 | 0.0074 | 0.0008 | 0.0024 | 0.0019 | 0.0033 | 0.0015 |
| Level 1 | ItemCF (Co-occurrence + Recency Decay $0.7^j$) | 0.1440 | 0.2767 | 0.1875 | 0.1541 | 0.1171 | 0.1100 | 0.1480 |
| Level 2 | GRU4Rec (Canonical PyTorch RNN) | 0.1161 | 0.2011 | 0.1525 | 0.1249 | 0.0934 | 0.0835 | 0.1199 |
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

## Core Observations

1. **ItemCF outperforms GRU4Rec on short sessions:** In sparse sessions ($77.8\%$ have $\le 2$ product interactions), statistical co-occurrence with recency decay outperforms a sequence RNN trained from scratch on 10.6k vocabulary items.
2. **Search Intent Deficit:** Both ItemCF and GRU4Rec show a massive drop in performance when search events occur ($0.1480 \to 0.1100$ and $0.1199 \to 0.0835$ respectively).
3. **Action Escalation Blindspot:** Both baselines collapse when predicting `add-to-cart` ($0.0065$ and $0.0039$) and `purchase` ($0.0181$ and $0.0192$).
