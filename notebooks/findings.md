# Dataset Findings (Coveo SIGIR eCOM 2021)

Empirical profiling conducted on `browsing_train.csv` (36.1M events), `search_train.csv` (819.5K events), and `sku_to_content.csv` (66.4K products).

## Session Structure & Cold-Start Distribution
* **Total unique sessions in 2M sample:** 274,724
* **0 product interactions:** 33.37% (91,671 sessions) — *Pure navigation, bounces, or non-product searches.*
* **1 product interaction:** 34.17% (93,860 sessions) — *Immediate departure after single view.*
* **2 product interactions:** 10.30% (28,307 sessions) — *Single input, single target.*
* **3+ product interactions:** 22.16% (60,886 sessions) — *Multi-step session.*
* **$\le 2$ product interactions:** **77.84%** of sessions.
* **Implication:** The recommender system operates under cold-start / sparse context for the vast majority of requests.

## Action Distribution (from 500k row sample)
* `pageview` (no SKU): **352,917 (70.6%)**
* `detail` (product view): **137,326 (27.5%)**
* `add` to cart: **4,513 (0.9%)** (present in only 4.3% of sessions)
* `remove`: **4,204 (0.8%)**
* `purchase`: **1,040 (0.2%)** (present in only 1.1% of sessions)

## Query & Product Vector Compatibility
* Query vectors (`search_train.csv`) and product description vectors (`sku_to_content.csv`) are both **50-dimensional vectors (`float32`)**.
* Mean L2 norm: Query $\approx 0.53$, Product $\approx 0.60$.
* Direct cross-product cosine similarity on clicked items has a mean of **$-0.0285$** (range: $-0.29$ to $+0.14$).
* **Finding:** Vectors are zero-centered representations from a projection space. A simple uncalibrated dot product is not an effective zero-shot ranker; alignment or supervision is necessary.

## Implications for Model Design
1. **Target Formulation:** Target must be the *next product interaction*, preserving action type `(target_item, target_action)`, while pageviews serve as contextual navigation signals.
2. **Cold-start priority:** Evaluated explicitly as `cold_start_1` and `cold_start_2`.
3. **Intent Opportunity:** The search gap ($0.1480 \to 0.1100$) and action gap ($0.1530 \to 0.0065$) provide clear targets for Phase 3 intent modeling.
