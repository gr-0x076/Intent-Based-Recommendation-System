# Project Plan: Intent-Based E-commerce Recommender System

> **Guiding Principle:** The central research question is:  
> *Does explicitly modeling user intent (beyond next-item pattern matching) produce measurably better recommendations?*  
> Every phase answers this question more precisely.

---

## First: What I Changed From Your Decisions (And Why)

Before the plan, let me be direct about where your thinking is correct, where I adjusted it, and why. Don't just accept anyone's plan blindly — understand each decision.

### ✅ What You Got Right

| Decision | Why It's Correct |
|---|---|
| Session = dataset-defined | Avoids inventing boundaries; stays scientifically honest |
| Latent intent, no fake labels | Critical. Forcing a fixed taxonomy on unlabeled data = garbage-in |
| ItemCF first, FAISS later | Matches how real production systems were built historically |
| LightGBM before neural ranker | Faster debugging, more interpretable ablations |
| Chronological session split | Most teams get this wrong; you got it right |
| Behavioral shift + demo cases for intent-shift eval | Honest about what you can and cannot claim |

### ⚠️ What I've Adjusted

**1. Dataset Risk — Coveo SIGIR 2021 needs a backup plan.**  
The Coveo dataset requires a form submission and approval. Approval can take days to weeks, or access may be restricted. If you wait for it before doing anything else, you lose your first sprint.  
**Fix:** Treat Coveo as primary. Start EDA on **OTTO Multi-Objective Recommendation Dataset** (Kaggle, instant access, more modern, similar schema: clicks/carts/orders/sessions) as the fallback and warm-up. It has no form gate.

**2. The sequential baseline is missing from your progression.**  
You wrote: *Popularity → ItemCF → Intent → Retrieval → Ranking*  
This is missing a crucial middle step. **Sequential models (GRU4Rec / SASRec) are between ItemCF and intent modeling** — they capture next-item patterns without explicit intent. If you skip this, you can't isolate what "intent" is adding.  
**Fix:** Popularity → ItemCF → **GRU4Rec (sequential)** → Intent-aware encoding → Retrieval → Ranker

**3. "What does one training example look like?" is not a question for later.**  
You raised this at the end of your document. It must be answered BEFORE Phase 1 begins. It is the hardest decision in the project. I've made it the first locked decision below.

**4. The demo architecture needs a real-time serving path designed upfront.**  
The interactive session simulator you described requires a live API that processes incoming events and returns updated recommendations. Without designing this early, the demo becomes a hardcoded slideshow at the end. I've added a serving design decision in Phase 0.

**5. Cold-start must be explicitly tested, not just mentioned.**  
Your PRD lists it as a success criterion. Your evaluation plan doesn't concretely address how to measure cold-start. I've added a dedicated cold-start evaluation protocol.

---

## The One Decision Locked Before Any Code

> **What is one training example?**

This is the single most important decision. Everything downstream (model architecture, feature extraction, evaluation harness, data pipeline) depends on it.

**Decision: Next-Event Prediction from Partial Sessions**

```
Input:  Session events up to position t   → [e₁, e₂, e₃, ..., eₜ]
Target: The item at position t+1          → item_id of eₜ₊₁

Where each event eᵢ = {
    item_id (if product interaction),
    query_vector (if search),
    action_type (view / detail / add / purchase),
    timestamp
}
```

**Generation strategy (sliding window):**
```
Session: A → B → C → D → E

Training examples generated:
  [A]           → predict B
  [A, B]        → predict C
  [A, B, C]     → predict D
  [A, B, C, D]  → predict E

Evaluation:
  Use last event of validation sessions as target
  Use all prior events as input
```

**Candidate set:**
- All items that appear in training sessions (not the full catalog)
- Size: ~50K–500K items depending on dataset

**Primary metric:** MRR@20  
**Secondary metrics:** Recall@20, NDCG@20  
**Segment metrics:** by session length, by action mix, by cold-start status

---

## Project Phases

```
Phase 0: Foundation          (Days 1–7)
Phase 1: Baselines           (Days 7–21)
Phase 2: Sequential Model    (Days 21–35)
Phase 3: Intent Encoding     (Days 35–49)
Phase 4: Retrieval + Ranking (Days 49–63)
Phase 5: Integration & Demo  (Days 63–77)
Phase 6: Evaluation & Docs   (Days 77–84)
```

Each phase has a **go/no-go gate** — a measurable condition that must be met before proceeding.

---

## Phase 0: Foundation (Days 1–7)

> **Goal:** Get the ground truth right before anything else. A wrong evaluation harness invalidates every experiment.

### What Gets Built

**0.1 Data Access (Day 1–2)**
- [ ] Submit Coveo SIGIR 2021 access form immediately
- [ ] While waiting: download **OTTO dataset from Kaggle** (instant, no form)
  - OTTO schema: `session_id`, `aid` (item), `ts` (timestamp), `type` (clicks/carts/orders)
  - Closest modern equivalent to Coveo; usable as development dataset
- [ ] Map OTTO schema to the same internal schema you'll use for Coveo

**0.2 Evaluation Harness (Day 2–4)**  
Build this BEFORE any model. It is the most important code in the project.

```python
# evaluation.py — build this first
def evaluate(model, sessions, k=20):
    """
    Input:  list of sessions, each = list of events in order
    For each session:
        - use events[:-1] as input
        - use events[-1].item_id as target
    Returns: MRR@k, Recall@k, NDCG@k
    """

def evaluate_by_segment(model, sessions, k=20):
    """
    Segments:
    - short sessions (len < 3), medium (3-10), long (>10)
    - cold-start (first 2 events only)
    - search-heavy sessions (>50% search events)
    - purchase-intent sessions (contains add-to-cart)
    """

def cold_start_eval(model, sessions, k=20):
    """Only use the first 1-2 events of each session as input"""
```

**0.3 Data Splitting (Day 3–4)**
```
Chronological split by session start time:

  Earliest 70% of sessions   → TRAIN
  Next 15% of sessions        → VALIDATION
  Latest 15% of sessions      → TEST

NEVER shuffle. NEVER split individual rows.
Within each session: preserve chronological order.
```

**0.4 EDA (Exploratory Data Analysis) (Day 4–6)**

Before modeling, understand what you have:
- Distribution of session lengths
- Ratio of event types (view / search / add / purchase)
- Item popularity distribution (long tail?)
- Average time between events
- Fraction of sessions with search queries
- How many sessions have add-to-cart events
- Cold-start: what fraction of sessions have ≤2 events?

Why this matters: EDA tells you what your models will actually face. If 80% of sessions have ≤3 events, cold-start is your primary problem. If only 5% of sessions have search queries, query-based intent modeling has limited coverage.

**0.5 Internal Schema Definition (Day 5–6)**

Lock this. Every module uses this schema.
```python
# Every event in the system is represented as:
Event = {
    "session_id": str,
    "event_idx": int,          # position in session (0-indexed)
    "timestamp_ms": int,
    "item_id": str | None,     # None for pure search events
    "action_type": str,        # "view", "detail", "add", "purchase", "search"
    "query_vector": list | None, # 128-dim float list, None if not a search
    "is_target": bool          # True only for the evaluation target event
}
```

**0.6 Demo Architecture Decision (Day 6–7)**

Decide the serving path NOW so the demo isn't an afterthought:
```
Frontend (Streamlit)
    ↕ HTTP
FastAPI server
    ├── session_state: in-memory dict (session_id → list of events)
    ├── POST /event  → append event to session, return updated recommendations
    └── GET /recommend/{session_id} → run model on current session, return top-20
```
This means you need a model that can run in <500ms for a demo. Keep this in mind when choosing sequential models.

### Phase 0 Gate ✅
- [ ] Evaluation harness produces numbers on dummy data
- [ ] EDA report written (can be a Jupyter notebook)
- [ ] Training example format agreed and documented
- [ ] Dataset split verified (no leakage confirmed)
- [ ] Either Coveo or OTTO data is loaded and queryable

---

## Phase 1: Baselines (Days 7–21)

> **Goal:** Establish the floor. You cannot know if intent modeling helps without a strong baseline to beat.

### What Gets Built

**1.1 Popularity Baseline**
```python
# Most popular items in training sessions
# Returns the same top-20 popular items for every session
# This is the dumbest possible recommender
```
Expected MRR@20: ~0.02–0.05 (it's bad, but it's your floor)

**1.2 Most-Recent Item Baseline**
```python
# Recommend items most similar to the last interacted item
# Uses only the last event in the session
# No history, no memory
```
Expected MRR@20: slightly better than popularity

**1.3 ItemCF (Co-occurrence Based)**  
This is adapted directly from established session-based co-occurrence heuristics's `multi_recall.py`.

Core algorithm:
```python
# For each item pair (i, j) that appear in the same session:
#   sim[i][j] += 1 / log(1 + session_length)   # normalize by session size

# Recommendation for session with events [e1, e2, e3]:
#   For each ei in recent events (most recent first):
#       Candidates from sim[ei] weighted by:
#         - position_weight = 0.7^(position from end)
#         - time_weight = 1 - (time_diff / max_time_diff)
#       Add weighted score to candidates dict
#   Return top-20 by score
```

- Build on baseline co-occurrence models code but adapt to your evaluation harness
- Track: MRR@20, Recall@20

**1.4 URL-based CF (from baseline co-occurrence models)**  
Only if Coveo data is available (URL field exists). Skip on OTTO.

**Baseline Results Table** (fill in during experiments):
```
Model               MRR@20    Recall@20    NDCG@20    Cold-start MRR
──────────────────────────────────────────────────────────────────────
Popularity          _____      _____        _____       _____
Most-Recent         _____      _____        _____       _____
ItemCF              _____      _____        _____       _____
URL-CF (Coveo only) _____      _____        _____       _____
```

### Phase 1 Gate ✅
- [ ] ItemCF MRR@20 is measurably better than popularity
- [ ] Numbers match expected ranges from literature (ItemCF ~0.10–0.18 on session data)
- [ ] Evaluation runs without data leakage (confirmed by checking session split)
- [ ] Cold-start degradation is measured (you'll see ItemCF drops badly with 1–2 events)

---

## Phase 2: Sequential Baseline (Days 21–35)

> **Goal:** Capture order and recency with a model that learns from sequences. This is the step your original plan skipped, and it's critical.

### Why This Phase Exists

ItemCF treats a session as a **bag of items** — order doesn't matter. But intent is inherently sequential:
- Browse → search → browse → add-to-cart reads very differently from add-to-cart → browse → search
- A sequential model can capture this. Intent modeling builds *on top* of sequential models.

Without a sequential baseline, you can't know if improvements from intent modeling come from the intent representation or just from the sequential encoding.

### What Gets Built

**2.1 GRU4Rec (Session-RNN)**

The simplest meaningful sequential model. Treats each session as a sequence fed into a GRU.

```python
class GRU4Rec(nn.Module):
    def __init__(self, n_items, hidden_size=128, n_layers=1):
        self.item_embedding = nn.Embedding(n_items, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.output_layer = nn.Linear(hidden_size, n_items)

    def forward(self, session_items):
        # session_items: [seq_len, batch_size]
        embedded = self.item_embedding(session_items)
        output, hidden = self.gru(embedded)
        # Use last hidden state → scores over all items
        scores = self.output_layer(output[-1])  # [batch, n_items]
        return scores
```

Training: Cross-entropy loss over all items, or BPR (Bayesian Personalized Ranking) loss.

**2.2 SASRec (Attention-Based) — Optional, if time permits**

Transformer-based sequential model. More expressive than GRU4Rec, but slower to train.
- Use the original SASRec codebase (available on GitHub) and adapt to your dataset
- Only run this if GRU4Rec shows significant improvement over ItemCF

**Key insight for Phase 2:**  
If GRU4Rec barely improves over ItemCF, it may indicate that:
- Sessions are too short for sequential patterns to emerge
- The dataset is too sparse
- Your training example generation needs adjustment

In that case, the intent model needs to work differently than you expected.

### Updated Results Table:
```
Model               MRR@20    Recall@20    Cold-start
──────────────────────────────────────────────────────
Popularity          _____      _____        _____
ItemCF              _____      _____        _____
GRU4Rec             _____      _____        _____
SASRec (optional)   _____      _____        _____
```

### Phase 2 Gate ✅
- [ ] GRU4Rec improves over ItemCF on at least one metric
- [ ] Training converges (loss decreases, not oscillating)
- [ ] Inference for one session takes <100ms (demo feasibility check)
- [ ] Cold-start behavior is measured — GRU4Rec likely still struggles here

---

## Phase 3: Intent Encoding Layer (Days 35–49)

> **Goal:** This is the core novel contribution. Replace "what item did you last see" with "what are you trying to achieve."

### The Design Question: What Is the Input to the Intent Encoder?

You have two possible intent signals:

| Signal | Available in Coveo | Available in OTTO | Richness |
|--------|-------------------|------------------|----------|
| **Item sequences** | ✅ | ✅ | Medium |
| **Search query vectors** | ✅ | ❌ | High (semantic!) |
| **Action types** (view/add/purchase) | ✅ | ✅ (as type field) | Medium |
| **Time gaps between events** | ✅ | ✅ | Low-Medium |

**If Coveo is available:** Use query vectors + item sequences together.  
**If OTTO is fallback:** Use item sequences + action type weights.

### What Gets Built

**3.1 Weighted Session Encoder**

A simple but meaningful improvement over GRU4Rec: weight input events by their **action type signal strength** before encoding.

```python
ACTION_WEIGHTS = {
    "view": 1.0,
    "detail": 1.5,
    "search": 2.0,      # search is a strong explicit intent signal
    "add": 3.0,         # very strong
    "purchase": 4.0     # strongest
}

def weighted_session_repr(events, model):
    """
    events: list of Event dicts
    Each event's embedding is multiplied by its action weight.
    Final representation = weighted average of all event embeddings.
    """
    embeddings = []
    weights = []
    for event in events:
        w = ACTION_WEIGHTS[event['action_type']]
        emb = model.encode(event)     # item embedding or query embedding
        embeddings.append(emb * w)
        weights.append(w)
    session_repr = sum(embeddings) / sum(weights)
    return session_repr
```

**3.2 Intent Vector from Queries (Coveo only)**

If you have search queries as 128-dim vectors, you can extract intent directly:

```python
def extract_query_intent(events):
    """
    Find all search events in session.
    Most recent search = strongest intent signal.
    If no searches: fall back to item embedding average.
    """
    search_events = [e for e in events if e['query_vector'] is not None]
    if search_events:
        # Recency-weighted average of query vectors
        n = len(search_events)
        weights = [0.7 ** (n - 1 - i) for i in range(n)]  # recent = higher weight
        intent_vec = np.average(
            [e['query_vector'] for e in search_events],
            weights=weights, axis=0
        )
        return intent_vec  # this IS the intent vector
    else:
        return None  # fall back to sequential model representation
```

**3.3 Multi-Intent Representation**

Instead of one intent vector, produce a distribution over K intent prototypes:

```python
# Concept: K-means cluster item embeddings into K clusters = K intent prototypes
# Each cluster centroid = one type of intent (browsing, purchase-ready, comparing, etc.)
# Session representation = soft assignment to each cluster

def compute_intent_distribution(session_repr, intent_prototypes, K=8):
    """
    session_repr: [dim] float vector
    intent_prototypes: [K, dim] — learned or k-means centroids
    Returns: [K] probability distribution over intents
    """
    similarities = cosine_similarity(session_repr, intent_prototypes)  # [K]
    intent_dist = softmax(similarities / temperature)  # [K]
    return intent_dist
```

The final recommendation can then blend candidates from each intent cluster, weighted by the distribution.

**3.4 Hybrid Intent Representation**

Combine sequential model output with query-derived intent:

```python
def hybrid_intent(session_events, gru4rec, query_intent):
    seq_repr = gru4rec.encode(session_events)   # [dim]
    q_intent = extract_query_intent(session_events)  # [dim] or None

    if q_intent is not None:
        alpha = 0.4  # blend weight — tune this
        intent = alpha * q_intent + (1 - alpha) * seq_repr
    else:
        intent = seq_repr
    return intent
```

### Key Experiment in Phase 3

Run an **ablation comparison**:

```
Model A: GRU4Rec                       (sequential only)
Model B: GRU4Rec + action weights      (weighted sequential)
Model C: Query intent only             (search signals only)
Model D: Hybrid intent (B + C)         (the full intent model)
```

This directly answers: **Does explicit intent modeling add value over sequential pattern matching?**

### Phase 3 Gate ✅
- [ ] Hybrid intent model improves over GRU4Rec by ≥5% MRR@20
- [ ] The improvement is larger on search-heavy sessions than item-only sessions (validates the intent hypothesis)
- [ ] Cold-start performance improves (the intent model should handle sparse sessions better)
- [ ] Ablation table is filled and shows which components contribute

---

## Phase 4: Retrieval + Ranking (Days 49–63)

> **Goal:** Turn the intent vector into a scalable candidate generation + re-ranking pipeline.

### What Gets Built

**4.1 Dense Retrieval with FAISS**

Pre-compute item embeddings. Use the intent vector to find nearest-neighbor items.

```python
import faiss

# Offline: pre-compute and index all item embeddings
item_embeddings = []
for item_id in all_items:
    emb = item_encoder(item_id)   # from item tower or simple embedding
    item_embeddings.append(emb)

item_embeddings = np.array(item_embeddings).astype('float32')

# Build FAISS index
index = faiss.IndexFlatIP(dim)   # inner product (= cosine if normalized)
faiss.normalize_L2(item_embeddings)
index.add(item_embeddings)

# Online: find nearest items to intent vector
def retrieve_candidates(intent_vec, k=200):
    intent_vec = intent_vec.astype('float32').reshape(1, -1)
    faiss.normalize_L2(intent_vec)
    scores, item_indices = index.search(intent_vec, k)
    return item_indices[0], scores[0]
```

**4.2 Merged Candidate Pool**

Combine candidates from ItemCF and dense retrieval:

```python
def get_candidates(session_events, k=200):
    # Source 1: ItemCF candidates (from Phase 1 code)
    itemcf_candidates = itemcf_recall(session_events, top_k=100)

    # Source 2: Dense intent retrieval (from FAISS)
    intent_vec = compute_intent_vector(session_events)
    dense_candidates, dense_scores = retrieve_candidates(intent_vec, k=100)

    # Merge
    all_candidates = set(itemcf_candidates) | set(dense_candidates)
    return list(all_candidates)  # typically 100–150 unique items
```

**4.3 LightGBM Re-ranker**

For each (session, candidate_item) pair, build features and score with LightGBM.

Feature categories:
```python
features = {
    # ItemCF signals
    "itemcf_score": float,          # score from ItemCF recall
    "itemcf_rank": int,             # rank in ItemCF list

    # Dense retrieval signals
    "dense_score": float,           # cosine similarity to intent vec
    "dense_rank": int,

    # Session-level features
    "session_length": int,
    "nb_search_events": int,
    "nb_add_events": int,
    "session_duration_ms": int,
    "last_event_type": category,

    # Item-level features
    "item_global_popularity": float,   # how often this item appears in training
    "item_category": category,
    "item_price_bucket": float,        # if Coveo; else skip

    # Intent similarity features
    "intent_cosine_sim": float,       # cosine(session_intent, item_emb)
    "intent_action_type_match": float, # does item match user's last action type?

    # Position in session
    "is_cold_start": bool,            # len(session_events) <= 2
}

# Label: 1 if this candidate was the actual next item, 0 otherwise
```

Training data generation:
```python
# For each training session, for each position t:
#   positive example = true next item
#   negative examples = K random candidates NOT equal to true next item
#   K typically = 100–1000 (importance sampling from popularity distribution)
```

### Phase 4 Gate ✅
- [ ] FAISS retrieval runs in <20ms for one session
- [ ] Merged candidate pool has >80% recall of true next items (candidates contain the answer)
- [ ] LightGBM re-ranker improves over using retrieval scores alone
- [ ] Full pipeline (retrieval + ranking) runs in <500ms total

---

## Phase 5: Integration & Demo (Days 63–77)

> **Goal:** Put everything behind an API and build the visual demo.

### What Gets Built

**5.1 FastAPI Backend**

```python
# main.py
from fastapi import FastAPI
app = FastAPI()

session_store = {}  # session_id → list of events (in-memory for demo)

@app.post("/session/{session_id}/event")
def add_event(session_id: str, event: EventPayload):
    """User performs an action. Append to session. Return new recommendations."""
    session_store.setdefault(session_id, []).append(event.dict())
    recs = recommender.predict(session_store[session_id], k=20)
    intent_info = recommender.get_intent_info(session_store[session_id])
    return {
        "recommendations": recs,
        "intent_vector_summary": intent_info  # for visualization
    }

@app.get("/session/{session_id}/recommendations")
def get_recommendations(session_id: str):
    """Get current recommendations without adding a new event."""
    ...
```

**5.2 Streamlit Demo**

```python
# demo.py
import streamlit as st
import requests

st.title("Intent-Based Recommender — Live Demo")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Session History")
    # Show scrollable list of events so far

    # Input controls
    action = st.selectbox("Perform action:", ["Search", "View Item", "Add to Cart"])
    if action == "Search":
        query = st.text_input("Enter search query:")
        if st.button("Search"):
            # Send to API → get new recommendations

with col2:
    st.subheader("Current Recommendations")
    # Show top-10 recommendations as cards

# Bottom panel: Intent visualization
st.subheader("How Your Intent Changed")
# Line chart: intent distribution over time (K intent dimensions)
# Show which intent cluster is currently dominant
```

**5.3 Comparison Panel (Key Demo Feature)**

Show side-by-side for the same session:
```
ItemCF Recommendations     |  Intent Model Recommendations
───────────────────────────|────────────────────────────────
[Based on viewed items]    |  [Based on inferred intent]
Item A                     |  Item D
Item B                     |  Item A
Item C                     |  Item E
...                        |  ...
```

This directly demonstrates whether and how intent modeling changes the output.

**5.4 Intent-Shift Demonstration Script**

Prepare a fixed scripted session sequence that visibly shows intent shift:

```
Script: "The Deliberate Shift"

Step 1: Search "casual sneakers" → View Sneaker A → View Sneaker B
  → Intent: casual footwear browsing
  → ItemCF shows: more sneakers
  → Intent model shows: more casual footwear variants

Step 2: Search "waterproof hiking shoes" → View Hiking Boot A
  → Intent shifts: outdoor/hiking
  → ItemCF shows: mix of sneakers + hiking (still weighted by old views)
  → Intent model shows: hiking/outdoor focused (responds faster to new signal)

Step 3: Add Hiking Boot A to cart
  → Intent: purchase-oriented
  → Intent model shows: complementary items (hiking socks, gaiters, poles)
  → ItemCF still shows similar boots
```

This scripted path is your primary demo. Run it every time. Judge will see the adaptation.

### Phase 5 Gate ✅
- [ ] API handles events and returns recommendations without crashing
- [ ] Demo runs end-to-end without errors
- [ ] The comparison panel shows visibly different outputs between models
- [ ] Intent visualization updates in real-time as events are added

---

## Phase 6: Evaluation & Documentation (Days 77–84)

> **Goal:** Produce the numbers and narrative that justify the system.

### What Gets Produced

**6.1 Ablation Study Table**

```
Model                          MRR@20  Recall@20  NDCG@20  Cold-start MRR
───────────────────────────────────────────────────────────────────────────
Popularity (floor)             ___     ___        ___      ___
ItemCF                         ___     ___        ___      ___
GRU4Rec                        ___     ___        ___      ___
Intent (action weights only)   ___     ___        ___      ___
Intent (query vectors only)    ___     ___        ___      ___
Intent (hybrid)                ___     ___        ___      ___
Intent + FAISS retrieval       ___     ___        ___      ___
Full system (+ LGB ranker)     ___     ___        ___      ___
```

**6.2 Segmented Analysis**

```
Segment                        MRR@20 (Intent)  MRR@20 (ItemCF)  Delta
──────────────────────────────────────────────────────────────────────
Short sessions (len ≤ 3)       ___              ___              ___
Medium sessions (4–10)         ___              ___              ___
Long sessions (>10)            ___              ___              ___
Search-heavy sessions           ___              ___              ___
No-search sessions              ___              ___              ___
Cold-start (1–2 events)        ___              ___              ___
```

If the intent model's biggest improvement is in **short sessions and search-heavy sessions**, that confirms your hypothesis: intent modeling adds the most value exactly where history-based approaches fail.

**6.3 Intent Shift Analysis**

Using behavioral heuristics (query category change, or action type shift from browse to add):
```
Sessions with detectable behavioral shift:   N = ____
Among these, recommendations shifted faster in intent model: ___% (vs ___% for ItemCF)
```

**6.4 Technical Documentation**
- Architecture diagram (use the one from this plan)
- Data processing pipeline description
- Model descriptions (one paragraph each)
- Evaluation protocol (so others can reproduce)
- Known limitations (honest section)

---

## Team Division (3-Person Team)

| Person | Ownership |
|--------|-----------|
| **Person A** | Data pipeline (Phases 0–1): EDA, session splits, evaluation harness, ItemCF baseline |
| **Person B** | Modeling (Phases 2–3): GRU4Rec, intent encoder, ablation experiments |
| **Person C** | System & Demo (Phases 4–5): FAISS, FastAPI, Streamlit, LightGBM ranker |

**Shared:** Weekly sync to align on evaluation harness (everyone uses the same `evaluate()` function).  
**Critical:** Person A finishes the evaluation harness by Day 4 so Persons B and C can start testing.

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Coveo dataset access delayed | High | High | Start on OTTO immediately; map schemas so switch is <1 day of work |
| GRU4Rec training is too slow | Medium | Medium | Reduce vocabulary size; use subset of training sessions; use GRU not LSTM |
| Intent model doesn't improve over ItemCF | Medium | High | This is a valid research result; document it; use ablation to explain why |
| Demo API too slow for real-time use | Medium | Medium | Pre-compute item embeddings; use cached session state |
| Coveo query vectors add no value | Low | Medium | Fall back to item-only intent; document finding |
| Team misaligned on evaluation format | High | High | Lock evaluation harness in Phase 0 before anyone writes a model |

---

## The Full Architecture (End State)

```
                          ┌──────────────────────────────────────────┐
                          │              User Session                 │
                          │  [search:"hiking shoes"]→[view:boot_A]   │
                          │  →[view:boot_B]→[add:boot_A]             │
                          └──────────────────┬───────────────────────┘
                                             │
                          ┌──────────────────▼───────────────────────┐
                          │           Intent Encoder                  │
                          │  - Action-weighted event embeddings        │
                          │  - Recency-weighted query intent           │
                          │  - Hybrid fusion (α·query + (1-α)·seq)    │
                          │  → Intent Vector [dim=128]                │
                          └───────┬───────────────────┬──────────────┘
                                  │                   │
               ┌──────────────────▼───┐   ┌───────────▼──────────────┐
               │    ItemCF Recall     │   │   FAISS Dense Retrieval   │
               │  (co-occurrence CF)  │   │  (ANN on intent vector)   │
               │  → top-100 items     │   │  → top-100 items          │
               └──────────────────┬───┘   └───────────┬──────────────┘
                                  │                   │
                          ┌───────▼───────────────────▼──────────────┐
                          │         Merged Candidate Pool             │
                          │         (~150-200 unique items)           │
                          └──────────────────┬───────────────────────┘
                                             │
                          ┌──────────────────▼───────────────────────┐
                          │         LightGBM Re-Ranker                │
                          │  Features: CF score + dense score +        │
                          │           intent similarity + popularity   │
                          │           + session features               │
                          └──────────────────┬───────────────────────┘
                                             │
                          ┌──────────────────▼───────────────────────┐
                          │         Top-20 Recommendations            │
                          └──────────────────────────────────────────┘
```

---

## The Central Research Narrative

Every phase is structured to answer one question more precisely:

```
Phase 1: "How good is the non-intent baseline?"
            → Establishes ItemCF performance ceiling

Phase 2: "How much does temporal ordering help?"
            → Isolates sequential signal value

Phase 3: "Does explicit intent modeling add value on top of sequential?"
            → Core research hypothesis tested here

Phase 4: "Can we scale this into a retrieval+ranking pipeline?"
            → Engineering validation

Phases 5–6: "Can we demonstrate this to a non-technical audience?"
             "How much does each component contribute?"
```

If Phase 3 shows the intent model adds ≥5% MRR, you have a **positive research result**.  
If Phase 3 shows no improvement, that is **also a research result** — and a valuable one. Document it, explain why, and your honest analysis will be respected.

---

## What You Start Tomorrow

1. **Submit the Coveo SIGIR 2021 access form** — right now
2. **Download the OTTO dataset from Kaggle** — takes 5 minutes, no gate
3. **Write the evaluation harness** — the `evaluate()` function above
4. **Run a quick EDA on OTTO** — understand session length distributions
5. **Lock and document the training example format** — everyone signs off

The first line of code you write should be `evaluate(model, sessions)`. Not a model. Not a pipeline. The thing that tells you if anything you build is actually working.

---

*"Build the thermometer before the furnace."*
