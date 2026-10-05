"""
Level 0 Baseline: Global Popularity Model.
Tracks global item interaction frequency across training sessions.
Returns top-K globally popular items.
"""

from collections import Counter
from typing import List, Dict, Any, Optional


class PopularityRecommender:
    def __init__(self, filter_seen: bool = True):
        self.item_counts = Counter()
        self.popular_items = []
        self.filter_seen = filter_seen

    def fit(self, train_sessions: Dict[str, List[Dict[str, Any]]]) -> None:
        """Counts frequency of all product interactions across training sessions."""
        self.item_counts.clear()
        for sid, events in train_sessions.items():
            for e in events:
                item_id = e.get("item_id")
                if item_id:
                    self.item_counts[item_id] += 1

        self.popular_items = [item for item, _ in self.item_counts.most_common()]
        print(f"--> [Popularity] Fitted on {len(train_sessions):,} sessions. Unique items tracked: {len(self.popular_items):,}")

    def predict(self, input_events: List[Dict[str, Any]], top_k: int = 20) -> List[str]:
        """Returns top-K popular items, optionally filtering out items already interacted with in input_events."""
        if not self.filter_seen:
            return self.popular_items[:top_k]

        seen_items = {e["item_id"] for e in input_events if e.get("item_id")}
        recommendations = []
        for item in self.popular_items:
            if item not in seen_items:
                recommendations.append(item)
            if len(recommendations) >= top_k:
                break
        return recommendations
