# IntentRec — Intent-Based E-Commerce Recommender System

> Recommending what users need right now, not just what they liked before.

IntentRec is a session-aware recommendation system that dynamically infers user intent from search queries and user interactions. Unlike traditional recommenders that rely heavily on long-term history, IntentRec focuses on understanding a user's current goal and adapting recommendations in real time.

Built using the Coveo SIGIR 2021 E-Commerce Dataset.

---

## Problem

Traditional recommendation systems often assume user preferences are stable.

In reality, user goals change rapidly.

Examples:

- A user who usually buys electronics may suddenly search for formal shoes.
- A user searching for gaming laptops today may need accessories tomorrow.
- A user researching Python tutorials may later focus on interview preparation.

IntentRec aims to model evolving user intent and generate recommendations that reflect the user's current objective.

---

## Approach

```text
Search Query + Session Events
                │
                ▼
      Query Embeddings
                │
                ▼
        GRU Session Encoder
                │
                ▼
       Intent Representation
                │
                ▼
         FAISS Retrieval
                │
                ▼
      Candidate Products
                │
                ▼
          Re-Ranking
                │
                ▼
       Final Recommendations
```

---

## Key Features

- Session-aware recommendation
- Query-aware intent modeling
- Dynamic intent representation
- Intent-shift adaptation
- FAISS-based retrieval
- Intent-aware ranking

---

## Dataset

### Coveo SIGIR 2021 Dataset

Contains:

- Search queries
- Product views
- Click events
- Cart additions
- Purchases
- Product metadata
- Session information

This combination of query and behavioral data makes it ideal for intent modeling.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Session Encoder | GRU |
| Text Embeddings | Sentence Transformers |
| Retrieval | FAISS |
| Backend | FastAPI |
| Database | PostgreSQL |
| Experiment Tracking | MLflow |
| Demo UI | Streamlit |

---

## Evaluation Metrics

The system will be evaluated using:

- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@K
- Diversity
- Intent Shift Performance

---

## Results

| Model | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---------|---------|---------|---------|---------|
| Popularity Baseline | - | - | - | - |
| Item-KNN | - | - | - | - |
| IntentRec | - | - | - | - |

---

## Repository Structure

```text
intentrec/
│
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   ├── ranking/
│   ├── evaluation/
│   └── api/
│
├── app/
├── experiments/
├── tests/
└── README.md
```

---

## Current Status

🟡 In Development

### Completed
- Project setup
- Dataset acquisition
- Initial data loading

### In Progress
- Dataset exploration
- Session analysis
- Baseline recommender implementation

### Upcoming
- Query embedding pipeline
- GRU session encoder
- FAISS retrieval system

---

## Future Improvements

- Multi-intent modeling
- Diversity-aware ranking (MMR)
- UMAP visualization
- Real-time feedback adaptation
- Docker deployment

---

## Team

**Tensor Titans**

Building recommendation systems that understand intent, not just history.
