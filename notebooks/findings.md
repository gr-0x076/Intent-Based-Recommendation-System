# Empirical Dataset Findings (Coveo E-Commerce Session Benchmark)

Profiling conducted across `browsing_train.csv` (36.1M events), `search_train.csv` (819.5K events), and `sku_to_content.csv` (66.4K products), validated on $N = 24,968$ prefix-expanded validation sessions.

## 1. Session Structure & Cold-Start Dominance
* **Total unique sessions in 2M sample:** 274,724
* **0 product interactions:** 33.37% (91,671 sessions) — *Pure navigation / bounces.*
* **1 product interaction:** 34.17% (93,860 sessions) — *Single view.*
* **2 product interactions:** 10.30% (28,307 sessions) — *Single input, single target.*
* **3+ product interactions:** 22.16% (60,886 sessions) — *Multi-step session.*
* **$\le 2$ product interactions:** **77.84%** of sessions.
* **Implication:** The recommender system operates under extreme cold-start / sparse context for nearly 80% of user visits.

## 2. Action Sparsity & Repeat-Item Distribution
* `pageview` (no SKU): **352,917 (70.6%)**
* `detail` (product view): **137,326 (27.5%)**
* `add` to cart: **4,513 (0.9%)**
* `remove`: **4,204 (0.8%)**
* `purchase`: **1,040 (0.2%)**

### Target Re-occurrence in Validation Set (N = 24,968)
* **`detail` targets:** 37.1% were previously seen in input; 14.8% were the exact last item viewed.
* **`add` targets:** **95.0%** were previously seen in input; **92.2%** were the exact last item viewed.
* **`purchase` targets:** **94.3%** were previously seen in input; **78.2%** were the exact last item viewed.
* **Sanity Result:** A simple `Repeat Last Item` baseline scores **0.9219 MRR on `add` targets** and **0.7824 MRR on `purchase` targets**.
* **Implication:** Recommendation requires distinct evaluation modes:
  * *Discovery Mode (Unseen targets):* `filter_seen=True`
  * *Conversion Mode (Repeat targets):* `filter_seen=False`

## 3. Stratified Search Analysis (Controlling for Length)
* **Confounder Identification:** 97.8% of searches occur in long sessions (`len_6_plus`).
* When controlling for length:
  * `len_3_to_5`: Search MRR = 0.0963 vs. No-Search MRR = 0.1957 ($\Delta = -0.0994$, $N_{\text{search}}=57$).
  * `len_6_plus`: Search MRR = 0.1103 vs. No-Search MRR = 0.1240 ($\Delta = -0.0138$, $N_{\text{search}}=2,554$).
* **Statistical Finding:** ItemCF performance remains lower on search sessions even after controlling for session length, but the effect is much smaller than the raw comparison suggests:
  $$\Delta \text{ MRR@20} = -0.0138, \quad 95\% \text{ bootstrap CI } [-0.0256, -0.0024]$$

## 4. Query Vector Characteristics
* Query vectors and product vectors share a 50-dimensional space, but empirical cosine similarity on clicked products is centered around zero (mean $\approx -0.0285$). Uncalibrated raw dot products do not work as zero-shot rankers; learned alignment or projection is necessary.
