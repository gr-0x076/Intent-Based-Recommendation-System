# Architecture

## Current Hypothesis

```text
Query + Session Events
        ↓
Representation Learning
        ↓
Intent Representation
        ↓
Candidate Retrieval
        ↓
Ranking
        ↓
Recommendations
```

## Important

This architecture is a research hypothesis, not the final implementation.

Components such as:

- GRU
- Sentence Transformers
- FAISS
- Multi-intent modeling

will be validated experimentally before becoming part of the final system.
