# Research Plan

## Hypothesis

Current-session behavior may contain information about user intent
that historical or purely item-based recommendation methods fail to capture.

## Baseline Ladder

1. Random
2. Popularity
3. Item-KNN / ItemCF
4. 2021 SIGIR eCOM-style classical approach
5. Sequential recommendation baseline
6. Intent-aware model
7. Intent-aware + ranking improvements

## Core Research Questions

RQ1. How strong are classical session-based methods?

RQ2. Does sequential modeling improve recommendation quality?

RQ3. Does explicit query/context information improve performance?

RQ4. Does intent-aware modeling help more during intent shifts?

RQ5. How does performance change for short/cold sessions?

## Evaluation

Recall@K
MRR
NDCG@K

Additional analysis:

- intent-shift sessions
- normal sessions
- short sessions
- sparse sessions
- cold-start situations
