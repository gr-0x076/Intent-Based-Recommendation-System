# Baseline Results & Research Findings

Evaluated on the exact same chronological validation set ($N = 24,968$ prefix-expanded session examples from 22,524 validation sessions).

## 1. Official Baseline Ladder

| Level | Model | MRR@20 | Recall@20 | Cold-1 MRR | Cold-2 MRR | Rich 3+ MRR | Has-Search | No-Search | Discovery MRR | Conversion MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Level 0 | Global Popularity (Filter Seen=True) | 0.0017 | 0.0074 | 0.0008 | 0.0024 | 0.0019 | 0.0033 | 0.0015 | 0.0028 | 0.0000 |
| Level 1 | ItemCF (Co-occurrence + Recency Decay $0.7^j$) | **0.1440** | **0.2767** | **0.1875** | **0.1541** | **0.1171** | **0.1100** | **0.1480** | **0.2489** | 0.0000 |
| Level 2 | Pure GRU4Rec (Item Sequences Only) | 0.1161 | 0.2011 | 0.1525 | 0.1249 | 0.0934 | 0.0835 | 0.1199 | 0.1944 | 0.0000 |
| Level 3A | **GRU + Raw Features (Control)** | **0.1216** | **0.2130** | **0.1609** | **0.1300** | **0.0974** | **0.0896** | **0.1253** | **0.2035** | 0.0000 |
| [Sanity] | Repeat Last Item (Recency Prior) | 0.1896 | 0.1896 | 0.2634 | 0.1871 | 0.1510 | 0.1923 | 0.1893 | 0.0000 | **0.4708** |
| Level 3B | GRU + Latent Intent Bottleneck | *Phase 3 Treatment* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 2. Empirical Discoveries & Diagnostic Analyses

### A. Phase 3 3-Way Ablation Milestone: Model 3A (Control) Established
* **Result:** Adding raw action embeddings (16-D) and search query projections (32-D) via standard concatenation improves GRU performance from **0.1161 to 0.1216 MRR@20** ($+4.7\%$ relative gain).
* **The Control Standard:** This establishes the required control baseline. When Model 3B (Latent Intent Bottleneck) is evaluated, the hypothesis test is **3B vs. 3A (0.1216)**. Only gains exceeding $0.1216$ can be scientifically attributed to intent abstraction rather than simple feature augmentation.

### B. Action-Target Gap & The Repeat-Item Discovery
Under discovery evaluation with seen-item filtering (`filter_seen=True`), ItemCF scored **0.0065 on add-to-cart** and **0.0181 on purchase**.

Our target sanity check revealed:
```
=== Repeat-Target Distribution in Validation Set (N = 24,968) ===
Action: detail    | Total: 23,368 | Seen in Input: 8,669 (37.1%) | Exactly Last Item: 3,464 (14.8%)
Action: add       | Total:  1,088 | Seen in Input: 1,034 (95.0%) | Exactly Last Item: 1,003 (92.2%)
Action: purchase  | Total:    193 | Seen in Input:   182 (94.3%) | Exactly Last Item:   151 (78.2%)
```
* **Discovery:** In 95.0% of add-to-cart events and 94.3% of purchases, the target item was already viewed in the session prefix.
* A trivial `Repeat Last Item` baseline achieves **0.9219 MRR on add-to-cart** and **0.7824 MRR on purchase**.
* **Implication:** The low score of ItemCF and GRU4Rec on commitment actions was an artifact of `filter_seen=True`. Evaluation must distinguish:
  * **Discovery Mode:** Target is an unseen item ($59.7\%$ of targets, ItemCF MRR = 0.2489).
  * **Conversion Mode:** Target is an already examined item ($40.3\%$ of targets, Repeat Prior MRR = 0.4708).

### C. Stratified Search-Session Analysis (Controlling for Confounders)
Unstratified data showed an apparent drop of $-0.0380$ MRR on search sessions ($0.1100$ vs $0.1480$). However, **97.8% of all searches occur in long sessions (`len_6_plus`)**, which naturally suffer from co-occurrence dilution.

Controlling for session length yields:
```
=== Stratified ItemCF MRR@20 by Input Length ===
Strata: len_2       | Search: N=0     MRR=0.0000 | No-Search: N=1,852  MRR=0.2150 | Delta: N/A
Strata: len_3_to_5  | Search: N=57    MRR=0.0963 | No-Search: N=5,135  MRR=0.1957 | Delta: -0.0994
Strata: len_6_plus  | Search: N=2,554 MRR=0.1103 | No-Search: N=15,366 MRR=0.1240 | Delta: -0.0138
```
* **Finding:** ItemCF performance remains lower on search sessions even after controlling for session length, but the effect is much smaller than the raw comparison suggests:
  $$\Delta \text{ MRR@20} = -0.0138, \quad 95\% \text{ bootstrap CI } [-0.0256, -0.0024]$$
* **Interpretation:** While statistically significant ($p < 0.01$), the gap is an empirical observation of lower retrieval performance on search-containing sessions, not a proven causal relationship.
