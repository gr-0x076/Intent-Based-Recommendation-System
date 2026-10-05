# Updated Plan — After Looking at Your Actual Data

> This document replaces assumptions with facts. Read it before the original plan.

---

## What the Data Actually Looks Like

Before changing anything, here are the real numbers from your dataset:

### File Sizes
| File | Rows | Size |
|------|------|------|
| `browsing_train.csv` | **36,079,308** | 6.4 GB |
| `search_train.csv` | **819,517** | 1.7 GB |
| `sku_to_content.csv` | **66,386** | 74 MB |

### Browsing Event Schema (confirmed from file)
```
session_id_hash, event_type, product_action, product_sku_hash, 
server_timestamp_epoch_ms, hashed_url
```
Real sample row:
```
20c458b8..., event_product, detail, d5157f8b..., 1550885210881, 7e4527ac...
```

### Action Type Distribution (from 500K rows sample)
| Action | Count | % |
|--------|-------|---|
| `pageview` (no product) | 352,917 | **70.6%** |
| `detail` | 137,326 | 27.5% |
| `add` | 4,513 | 0.9% |
| `remove` | 4,204 | 0.8% |
| `purchase` | 1,040 | **0.2%** |

### Session Length Distribution (from 2M rows — 274,725 sessions)
| Sessions with ≤ N events | Count | % |
|--------------------------|-------|---|
| ≤ 1 event | 51,495 | **18.7%** |
| ≤ 2 events | 131,289 | **47.8%** |
| ≤ 3 events | 146,499 | **53.3%** |
| ≤ 5 events | 180,673 | **65.8%** |
| ≤ 10 events | 224,467 | 81.7% |
| ≤ 20 events | 253,154 | 92.1% |

### Session Signal Coverage
| Signal | Sessions | % |
|--------|----------|---|
| Has `detail` view | 181,933 | 66.2% |
| Has `add` to cart | 11,702 | **4.3%** |
| Has `purchase` | 2,923 | **1.1%** |
| Has search event | **550,100** | — |

### Product Catalog Coverage (`sku_to_content.csv`)
| Field | Products with data | % |
|-------|-------------------|---|
| `description_vector` | 31,950 | **48.1%** |
| `image_vector` | 28,370 | **42.7%** |
| `price_bucket` | 32,038 | 48.3% |
| `category_hash` | 32,052 | 48.3% |
| **No metadata at all** | **34,334** | **51.7%** |

### Search Data
- **819,516 search events** across **550,100 sessions**
- Each search has: `query_vector` (50-dim float), `clicked_skus_hash`, `product_skus_hash` (shown results)
- Query vector is a real 50-dimensional dense vector (confirmed from sample)

---

## What These Numbers Actually Mean for the Project

This is the important part. These numbers change several decisions.

### Finding 1: Cold-start is not an edge case — it IS the dominant case

> **53% of sessions have ≤ 3 events. 48% have ≤ 2 events.**

This means your model will spend more time in "cold-start" mode than in "rich history" mode. This is a **gift for the intent modeling angle** — it's precisely where search query signals matter most and where traditional ItemCF fails hardest.

**What changes:** Cold-start isn't a "stretch goal" or special segment. It is the primary use case. Design for it from Day 1.

### Finding 2: Purchase and add-to-cart signals are extremely rare

> **Only 4.3% of sessions have any add-to-cart event. Only 1.1% have a purchase.**

This means:
- For the **recommendation task**: you cannot use purchase/add as strong training signals because they barely exist in most sessions. Your training signal is primarily `detail` views and `pageviews`.
- For the **cart intent task** (`code_cart`): that 4.3% add-to-cart is actually a meaningful subset (~12K sessions from 2M). This is your positive class.
- **The dominant pattern in a session is: pageview → detail → pageview → detail**. Not add → purchase.

**What changes:** Don't over-engineer the action weighting. `detail` + `pageview` are your workhorses, not `add`/`purchase`.

### Finding 3: Search query vectors are gold — but coverage is limited

> **550,100 sessions have search data**, but browsing has ~5M sessions total.

That means only **~10–11% of sessions have any search signal**. This is not "we can build an intent model from search queries" — it's "search queries are a powerful feature when available, but your model must degrade gracefully without them for 90% of sessions."

**What changes:** The intent encoder must have two paths:
- **Path A (with search):** Use query vectors as the primary intent signal
- **Path B (without search):** Use `detail` view sequences as the intent proxy

### Finding 4: Product metadata is half empty

> **51.7% of products have NO metadata at all** (no description, no image, no price, no category).

This kills the idea of using rich content-based item representations as a core feature. You can use metadata as an optional enrichment, but your item representation must work on item ID alone.

**What changes:** Item embeddings must be learned from interaction co-occurrence (like ItemCF), not from product metadata. Use metadata as an optional feature in the LightGBM ranker, not as the item encoder input.

### Finding 5: The dataset is NOT sorted chronologically within the file

From the official documentation:
> "the order of the events in the file is not strictly chronological — refer to the session identifier and the timestamp information to reconstruct the actual chain of events for a given session."

**What changes:** Every pipeline step that reads sessions must sort by `server_timestamp_epoch_ms` within each session group. This is not optional.

### Finding 6: The dataset is huge — you cannot load it all into RAM

> 36M rows × 6.4GB. A pandas `read_csv` of the full file will likely use 15–25GB of RAM.

**What changes:** You need a chunked or sampled loading strategy. The plan below recommends working with a 10% stratified sample for development and the full dataset only for final training.

---

## Plan Changes (Delta from Original Plan)

Here is specifically what changes in the [Project_Plan.md], and why:

### Change 1: Add a Data Engineering Sprint Before Phase 1

Before any model work, you need a reliable, memory-efficient data pipeline. This is now **Phase 0B** (between EDA and baselines).

### Change 2: Reframe the evaluation target

**Original plan:** Predict "next item" from session prefix  
**Actual data constraint:** 70% of events are `pageview` (no product). You cannot predict the next event if it's a pageview.

**Corrected target:**  
> Predict the **next product interaction** (detail/add/purchase) given all events so far (including pageviews as context).

This means: filter out pageviews from the prediction target, but keep them as input features (they tell you which page/URL the user was on).

### Change 3: Cold-start is the primary evaluation segment, not a secondary one

Sessions with ≤ 3 product interactions = **majority of your dataset**. Structure the evaluation harness around this:

```python
SEGMENTS = {
    "cold_start":    lambda s: n_product_events(s) <= 2,
    "medium":        lambda s: 3 <= n_product_events(s) <= 8,
    "rich":          lambda s: n_product_events(s) > 8,
    "has_search":    lambda s: any(e['type'] == 'search' for e in s),
    "no_search":     lambda s: all(e['type'] != 'search' for e in s),
    "has_add":       lambda s: any(e['action'] == 'add' for e in s),
}
```

### Change 4: Two separate training tasks, not one

Given the data structure, you actually have two distinct modeling problems:

| Task | Input | Target | Sessions |
|------|-------|--------|----------|
| **Next-item recommendation** | Session events so far | Next product_sku_hash | ~5M sessions |
| **Cart-to-purchase prediction** | Events up to and including add | Will they purchase? | ~250K sessions (4.3%) |

These should be trained and evaluated separately. The plan keeps them together but now explicitly separates the data pipelines.

### Change 5: Working dataset size

Don't train on 36M rows to start. Use a **stratified 10% sample** (~3.6M rows, ~500K sessions) for development. Scale to full data only for final training.

```python
# How to sample: take 10% of unique sessions, keep all their events
import pandas as pd
df = pd.read_csv('browsing_train.csv', usecols=['session_id_hash'])
sessions = df['session_id_hash'].unique()
sample_sessions = pd.Series(sessions).sample(frac=0.1, random_state=42)
# Then filter full file for only these sessions
```

---

## Updated Phase Structure

```
Phase 0A: Data Understanding (Days 1–3)      ← EDA (largely done by reading this doc)
Phase 0B: Data Pipeline (Days 3–6)           ← NEW: chunked loading, sampling, schema
Phase 0C: Evaluation Harness (Days 6–8)      ← same as before, but with corrected target
Phase 1:  Baselines (Days 8–18)              ← Popularity + ItemCF on sample data
Phase 2:  Sequential Model (Days 18–30)      ← GRU4Rec on item sequences
Phase 3:  Intent Encoding (Days 30–44)       ← Dual-path: query-based + sequence-based
Phase 4:  Retrieval + Ranking (Days 44–56)   ← FAISS + LightGBM
Phase 5:  Integration & Demo (Days 56–68)    ← FastAPI + Streamlit
Phase 6:  Evaluation & Docs (Days 68–75)     ← Ablations, write-up
```

---

## Phase 0B: Data Pipeline (New — Most Important Phase)

> **Goal:** Build a memory-efficient, reusable data loading system. Everything else depends on this.

### Step 1: Session-level sampler

```python
# data/sampler.py
import pandas as pd
import pickle

def create_session_sample(browsing_path, search_path, 
                          sample_frac=0.1, seed=42, output_dir='data/processed/'):
    """
    Reads browsing_train.csv in chunks.
    Selects sample_frac of sessions.
    Saves sampled sessions as parquet files (much faster to reload).
    """
    # Step 1: Get all unique session IDs (read only that column — fast)
    print("Reading session IDs...")
    session_chunks = pd.read_csv(browsing_path, usecols=['session_id_hash'], 
                                  chunksize=1_000_000)
    all_sessions = set()
    for chunk in session_chunks:
        all_sessions.update(chunk['session_id_hash'].unique())
    
    # Step 2: Sample sessions
    all_sessions = list(all_sessions)
    n_sample = int(len(all_sessions) * sample_frac)
    import random; random.seed(seed)
    sampled_sessions = set(random.sample(all_sessions, n_sample))
    print(f"Sampled {n_sample:,} sessions from {len(all_sessions):,}")
    
    # Step 3: Filter and save browsing data
    print("Filtering browsing data...")
    filtered_chunks = []
    for chunk in pd.read_csv(browsing_path, chunksize=1_000_000):
        filtered = chunk[chunk['session_id_hash'].isin(sampled_sessions)]
        filtered_chunks.append(filtered)
    
    browsing_df = pd.concat(filtered_chunks)
    browsing_df.columns = ['session_id', 'event_type', 'action', 
                            'item_id', 'timestamp_ms', 'url_hash']
    browsing_df.to_parquet(f'{output_dir}browsing_sample.parquet')
    
    # Step 4: Filter and save search data
    search_df = pd.read_csv(search_path)
    search_df = search_df[search_df['session_id_hash'].isin(sampled_sessions)]
    search_df.columns = ['session_id', 'query_vector', 
                          'clicked_skus', 'result_skus', 'timestamp_ms']
    search_df.to_parquet(f'{output_dir}search_sample.parquet')
    
    print("Done. Files saved to", output_dir)
    return browsing_df, search_df
```

**Expected output:** ~360MB parquet file (10% of 3.6GB browsing) that loads in <5 seconds.

### Step 2: Session builder

```python
# data/session_builder.py

def build_sessions(browsing_df, search_df):
    """
    Takes the raw browsing + search dataframes.
    Returns a list of sessions, each session = list of events sorted by timestamp.
    Each event is a unified dict regardless of source (browsing vs search).
    """
    sessions = {}
    
    # Add browsing events
    for _, row in browsing_df.iterrows():
        sid = row['session_id']
        event = {
            'type': 'search' if pd.isna(row['item_id']) and not pd.isna(row.get('query_vector')) else 'product',
            'action': row['action'] if row['action'] else 'pageview',
            'item_id': row['item_id'],   # None for pageviews
            'url_hash': row['url_hash'],
            'timestamp_ms': row['timestamp_ms'],
            'query_vector': None,
        }
        sessions.setdefault(sid, []).append(event)
    
    # Add search events (merge into same session list)
    import ast
    for _, row in search_df.iterrows():
        sid = row['session_id']
        qv = ast.literal_eval(row['query_vector']) if isinstance(row['query_vector'], str) else None
        event = {
            'type': 'search',
            'action': 'search',
            'item_id': None,
            'url_hash': None,
            'timestamp_ms': row['timestamp_ms'],
            'query_vector': qv,          # 50-dim list
            'clicked_skus': ast.literal_eval(row['clicked_skus']) if row['clicked_skus'] else [],
            'result_skus': ast.literal_eval(row['result_skus']) if row['result_skus'] else [],
        }
        sessions.setdefault(sid, []).append(event)
    
    # Sort each session by timestamp
    for sid in sessions:
        sessions[sid].sort(key=lambda e: e['timestamp_ms'])
    
    return sessions  # dict: session_id → list of events
```

### Step 3: Chronological train/val/test split

```python
# data/splitter.py

def chronological_split(sessions, train_frac=0.7, val_frac=0.15):
    """
    Split sessions by their START timestamp (first event timestamp).
    Returns train_sessions, val_sessions, test_sessions dicts.
    """
    # Get start time of each session
    session_start_times = {
        sid: events[0]['timestamp_ms'] 
        for sid, events in sessions.items()
    }
    
    # Sort sessions by start time
    sorted_sessions = sorted(session_start_times.items(), key=lambda x: x[1])
    n = len(sorted_sessions)
    
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_ids = {sid for sid, _ in sorted_sessions[:train_end]}
    val_ids = {sid for sid, _ in sorted_sessions[train_end:val_end]}
    test_ids = {sid for sid, _ in sorted_sessions[val_end:]}
    
    return (
        {sid: sessions[sid] for sid in train_ids},
        {sid: sessions[sid] for sid in val_ids},
        {sid: sessions[sid] for sid in test_ids},
    )
```

### Step 4: Training example generator

```python
# data/examples.py

def generate_training_examples(sessions):
    """
    From each session, generate next-product-interaction prediction examples.
    
    Input: all events up to position t (inclusive)
    Target: the item_id of the NEXT product interaction after t
    
    Skips pageviews as targets (they have no item_id).
    Keeps pageviews in input context.
    """
    examples = []
    
    for sid, events in sessions.items():
        # Find indices of product events (have item_id)
        product_indices = [i for i, e in enumerate(events) 
                           if e['item_id'] is not None]
        
        # Need at least 2 product interactions to form one example
        if len(product_indices) < 2:
            continue
        
        # Generate examples: for each product event after the first,
        # use all events up to (but not including) it as input
        for k in range(1, len(product_indices)):
            target_idx = product_indices[k]
            target_item = events[target_idx]['item_id']
            
            # Input: all events before target (including pageviews as context)
            input_events = events[:target_idx]
            
            examples.append({
                'session_id': sid,
                'input_events': input_events,
                'target_item_id': target_item,
                'target_action': events[target_idx]['action'],
                'n_prior_product_events': k,  # for cold-start segmentation
                'has_search': any(e['type'] == 'search' for e in input_events),
            })
    
    return examples
```

### Phase 0B Gate ✅
- [ ] `browsing_sample.parquet` and `search_sample.parquet` exist and load in <10s
- [ ] `build_sessions()` produces correctly time-ordered events per session
- [ ] `chronological_split()` confirmed: no session appears in both train and val
- [ ] `generate_training_examples()` produces at least 100K examples from the 10% sample
- [ ] Sample data: confirm average session length, search coverage %, action distribution matches the EDA numbers above

---

## What Does NOT Change in the Plan

The rest of the plan (Phases 1–6) is structurally sound. The key thing you now know:

1. **Your real baseline number** for ItemCF will be modest — short sessions are hard
2. **Query vectors are 50-dimensional** (not 128 or 768 — simpler than expected, but real)
3. **Product metadata covers only ~48% of items** — lean on collaborative signals
4. **The "intent shift" demo** should feature a session with a search event (10% of sessions) — make the demo always start with a search to make the intent signal visible

---

## The Five Things to Do Right Now

In order. Don't skip any.

**1. Create the project workspace**
```bash
mkdir -p ~/intent-rec/{data/raw,data/processed,src,notebooks,models,api,demo}
ln -s /home/mohith/Downloads/SIGIR-ecom-data-challenge/train ~/intent-rec/data/raw/
```

**2. Run the full session sampler** (takes ~15–20 min on 36M rows)
```bash
cd ~/intent-rec
python src/data/sampler.py  # creates data/processed/browsing_sample.parquet
```

**3. Build sessions and confirm the split**
```bash
python src/data/session_builder.py
python src/data/splitter.py
# Should output: train=X sessions, val=Y sessions, test=Z sessions
```

**4. Write and run the evaluation harness on dummy output**
```bash
python src/evaluation.py --dummy
# Should output MRR@20 = some random low number, confirming the harness works
```

**5. Run ItemCF on the training sessions, evaluate on val**
```bash
python src/models/itemcf.py
# This is your first real number. Everything you build is judged against it.
```

---

## Revised "One Training Example" Definition

Updated based on real data:

```python
# ONE TRAINING EXAMPLE:
{
    "session_id": "abc123",
    
    "input_events": [
        # ALL events before the target (pageviews included as URL context):
        {"type": "pageview", "action": "pageview", "item_id": None, "url_hash": "url1", "timestamp_ms": T0},
        {"type": "search",   "action": "search",   "item_id": None, "query_vector": [...50 floats...], "timestamp_ms": T1},
        {"type": "product",  "action": "detail",   "item_id": "item_A", "url_hash": "url2", "timestamp_ms": T2},
        {"type": "pageview", "action": "pageview",  "item_id": None, "url_hash": "url3", "timestamp_ms": T3},
    ],
    
    "target_item_id": "item_B",     # next product the user interacted with
    "target_action": "detail",      # what they did with it (detail/add/purchase)
    "n_prior_product_events": 1,    # number of product interactions in input (cold-start = 1 or 2)
    "has_search": True,             # does input contain a search event?
}
```

This is the ground truth format. Every model receives `input_events`, predicts a ranked list, and is scored on whether `target_item_id` appears in the top-20.

---

## Key Numbers to Keep in Mind

```
~5M total sessions in browsing data
~550K sessions have search events (~11%)
~48% of sessions have ≤ 2 product interactions (cold-start)
~66K unique products (only 48% have any metadata)
Query vectors: 50-dimensional
Product description/image vectors: same 50-dim space (compatible!)
Dominant action: pageview (70%), then detail (27%), then add (<1%)
```

The last point about vector compatibility is significant:
> **Query vectors and product description vectors are in the same 50-dim embedding space.**

This means: you can directly compute cosine similarity between a search query vector and a product description vector. This IS your intent matching function for sessions with search events. No training needed for this part — it's given to you.

```python
# If user searched with query_vector q, and product has description_vector d:
intent_match_score = cosine_similarity(q, d)  # directly meaningful!
```

This is an extremely valuable feature that many teams at the competition likely underused.
