"""
Level 1 Baseline: Item-based Collaborative Filtering (ItemCF) with Position Decay.
Adapted from SIGIR eCOM 2021 1st-place core principles:
1. Item-to-item co-occurrence matrix normalized by session length.
2. Exponential recency decay (0.7^j) giving higher weight to immediate recent items.
3. Fallback to global popularity for cold-start / sparse items.
4. Ban filter suppressing items already seen in current session input.
"""

import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional


class ItemCFRecommender:
    def __init__(self, top_similar_items: int = 100, recency_decay: float = 0.7, filter_seen: bool = True):
        self.top_similar_items = top_similar_items
        self.recency_decay = recency_decay
        self.filter_seen = filter_seen

        # sim_matrix[item_a][item_b] = similarity
        self.sim_matrix = defaultdict(lambda: defaultdict(float))
        self.popular_items = []

    def fit(self, train_sessions: Dict[str, List[Dict[str, Any]]]) -> None:
        """Computes co-occurrence similarity matrix from training sessions."""
        print("--> [ItemCF] Building co-occurrence matrix from training sessions...")
        item_counts = Counter()

        for sid, events in train_sessions.items():
            # Extract chronological product sequence (deduplicating consecutive views)
            prod_seq = []
            for e in events:
                item_id = e.get("item_id")
                if item_id:
                    item_counts[item_id] += 1
                    if not prod_seq or prod_seq[-1] != item_id:
                        prod_seq.append(item_id)

            if len(prod_seq) < 2:
                continue

            session_weight = 1.0 / math.log(1.0 + len(prod_seq))

            # Co-occurrence across all pairs in session
            for i, item_i in enumerate(prod_seq):
                for j, item_j in enumerate(prod_seq):
                    if item_i == item_j:
                        continue
                    # Distance decay within session
                    dist = abs(i - j)
                    dist_weight = 1.0 / (1.0 + 0.5 * dist)
                    self.sim_matrix[item_i][item_j] += session_weight * dist_weight

        # Normalization: Cosine-like denominator sqrt(count_i * count_j)
        print("--> [ItemCF] Normalizing similarities...")
        for item_i, neighbors in self.sim_matrix.items():
            count_i = item_counts[item_i]
            for item_j, weight in neighbors.items():
                count_j = item_counts[item_j]
                self.sim_matrix[item_i][item_j] = weight / math.sqrt(count_i * count_j)

            # Prune to top-N neighbors to keep memory minimal and inference fast
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:self.top_similar_items]
            self.sim_matrix[item_i] = dict(sorted_neighbors)

        self.popular_items = [item for item, _ in item_counts.most_common()]
        print(f"--> [ItemCF] Fitted on {len(train_sessions):,} sessions. Unique items with similarity: {len(self.sim_matrix):,}")

    def predict(self, input_events: List[Dict[str, Any]], top_k: int = 20) -> List[str]:
        """
        Recommends top-K items based on recent product interactions in input_events.
        Applies exponential recency decay 0.7^j on recent items.
        """
        seen_items = set()
        product_interactions = []
        for e in input_events:
            item_id = e.get("item_id")
            if item_id:
                product_interactions.append(item_id)
                seen_items.add(item_id)

        candidate_scores = defaultdict(float)

        # Iterate from most recent product backwards
        recent_products = product_interactions[::-1]
        for j, seed_item in enumerate(recent_products[:15]):  # look back up to 15 items
            recency_w = max(0.1, self.recency_decay ** j)
            if seed_item in self.sim_matrix:
                for cand, sim_val in self.sim_matrix[seed_item].items():
                    if self.filter_seen and cand in seen_items:
                        continue
                    candidate_scores[cand] += sim_val * recency_w

        # Rank candidates by accumulated score
        ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item for item, _ in ranked_candidates[:top_k]]

        # Fallback to global popularity if candidates < top_k
        if len(recommendations) < top_k:
            for item in self.popular_items:
                if (not self.filter_seen or item not in seen_items) and item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) >= top_k:
                    break

        return recommendations[:top_k]
