# Comparative Survey of Session-Based Recommendation Paradigms

A systematic analysis of recommendation architectures for short-session e-commerce environments, comparing classical heuristics, neural sequence models, and intent-aware frameworks.

---

## 1. Classical Collaborative Filtering vs. Session-Based Recommendations

### Traditional User-Item CF
Traditional collaborative filtering (e.g., Matrix Factorization, User-KNN) assumes persistent user identities across weeks or months. In e-commerce, this assumption fails because:
* Most visitors browse anonymously or as first-time guests without login identities.
* User preferences are not static: a user shopping for running shoes on Monday may shop for kitchenware on Thursday.
* Historical profiles overfit past purchases (e.g., recommending washing machines after a user already purchased one).

### Session-Based ItemCF
Session-based ItemCF operates strictly within the current session boundary:
* Constructs an item-to-item co-occurrence graph across sessions.
* Down-weights co-occurrences in long, noisy sessions using inverse-log session length normalization:
  $$w_{\text{session}} = \frac{1}{\log(1 + |S|)}$$
* Applies exponential recency decay $0.7^j$ to prioritize immediate recent items over older interactions.
* **Advantage:** Highly robust, fast, and strong on ultra-short sessions.
* **Limitation:** Blind to keyword intent, search events, and behavioral commitment stages.

---

## 2. Sequential Neural Architectures (GRU4Rec / SASRec)

### GRU4Rec (Recurrent Neural Networks)
GRU4Rec models the sequence of product interactions as a Markov decision process using Gated Recurrent Units (GRU):
* Maps items into dense embedding vectors.
* Updates recurrent hidden state $h_t = \text{GRU}(h_{t-1}, e_t)$.
* Scores candidate items via output projection logits.
* **Findings in Sparse Regimes:** When the majority of sessions have $\le 2$ interactions, the recurrent state does not receive sufficient transition context, often trailing well-tuned co-occurrence models on early-session cold-start.

### Self-Attention (SASRec)
* Uses multi-head self-attention to assign dynamic weights to all prior items regardless of distance.
* Excels when sessions are moderately dense ($>5$ items), but requires substantial parameter tuning and negative sampling strategies to prevent overfitting on sparse catalogs.

---

## 3. The Need for Explicit Intent Modeling

Neither co-occurrence nor pure sequence modeling accounts for:
1. **Search Query Intent:** When a user searches for a specific keyword or attribute, their immediate goal shifts away from their prior browsing context.
2. **Action Progression:** An `add-to-cart` or `detail` view carries vastly different intent weight than a generic category `pageview`.
3. **Multi-Intent Journeys:** A user exploring alternatives versus a user ready to checkout exhibit distinct behavioral velocities and dwell patterns.

This motivates our **Intent-Based Recommender System**, combining sequential signals, search query embeddings, and behavioral action states into a unified adaptive recommendation engine.
