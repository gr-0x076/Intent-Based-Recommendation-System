"""
Evaluation metrics and harness for Intent-Based Session Recommender.
Implements:
- MRR@K (Mean Reciprocal Rank)
- Recall@K
- Granular Segmented reporting:
  * Prior product count (cold_start_1, cold_start_2, rich_3plus)
  * Search presence (has_search, no_search)
  * Target action type (target_detail, target_add, target_purchase)
  * Dual Evaluation Modes:
    - discovery_unseen: target item NOT seen in input history
    - conversion_repeat: target item WAS seen in input history
"""

from typing import List, Dict, Any, Optional
import numpy as np


def compute_reciprocal_rank(predictions: List[str], target_item: str, k: int = 20) -> float:
    """Computes Reciprocal Rank (1/rank) if target is in top-K predictions, else 0.0."""
    top_k = predictions[:k]
    try:
        idx = top_k.index(target_item)
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0


def compute_hit(predictions: List[str], target_item: str, k: int = 20) -> float:
    """Computes Hit (1.0 if target in top-K, else 0.0)."""
    return 1.0 if target_item in predictions[:k] else 0.0


class EvaluationHarness:
    def __init__(self, k: int = 20):
        self.k = k

    def evaluate(self, model: Any, test_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a model over a list of test examples.
        Each example dict must contain:
          - 'input_events': list of events up to target
          - 'target_item_id': str
          - 'target_action': str (detail/add/purchase)
          - 'n_prior_products': int
          - 'has_search': bool
        """
        assert len(test_examples) > 0, "No test examples provided"

        all_rr = []
        all_hits = []

        segments = {
            "cold_start_1": {"rr": [], "hits": []},      # exactly 1 prior product interaction
            "cold_start_2": {"rr": [], "hits": []},      # exactly 2 prior product interactions
            "rich_3plus":   {"rr": [], "hits": []},      # 3 or more prior product interactions
            "has_search":   {"rr": [], "hits": []},      # input has >=1 search event
            "no_search":    {"rr": [], "hits": []},      # input has 0 search events
            "discovery_unseen":  {"rr": [], "hits": []}, # target item not in input history
            "conversion_repeat": {"rr": [], "hits": []}, # target item was in input history
            "target_detail":     {"rr": [], "hits": []},
            "target_add":        {"rr": [], "hits": []},
            "target_purchase":   {"rr": [], "hits": []},
        }

        for ex in test_examples:
            target = ex["target_item_id"]
            preds = model.predict(ex["input_events"], top_k=self.k)

            rr = compute_reciprocal_rank(preds, target, self.k)
            hit = compute_hit(preds, target, self.k)

            all_rr.append(rr)
            all_hits.append(hit)

            # Prior product counts
            n_p = ex.get("n_prior_products", 0)
            if n_p == 1:
                segments["cold_start_1"]["rr"].append(rr)
                segments["cold_start_1"]["hits"].append(hit)
            elif n_p == 2:
                segments["cold_start_2"]["rr"].append(rr)
                segments["cold_start_2"]["hits"].append(hit)
            else:
                segments["rich_3plus"]["rr"].append(rr)
                segments["rich_3plus"]["hits"].append(hit)

            # Search presence
            if ex.get("has_search", False):
                segments["has_search"]["rr"].append(rr)
                segments["has_search"]["hits"].append(hit)
            else:
                segments["no_search"]["rr"].append(rr)
                segments["no_search"]["hits"].append(hit)

            # Dual-mode: Discovery (unseen) vs Conversion (repeat)
            seen_items = {e["item_id"] for e in ex["input_events"] if e.get("item_id")}
            if target in seen_items:
                segments["conversion_repeat"]["rr"].append(rr)
                segments["conversion_repeat"]["hits"].append(hit)
            else:
                segments["discovery_unseen"]["rr"].append(rr)
                segments["discovery_unseen"]["hits"].append(hit)

            # Action breakdown
            action = ex.get("target_action", "detail")
            act_key = f"target_{action}"
            if act_key in segments:
                segments[act_key]["rr"].append(rr)
                segments[act_key]["hits"].append(hit)

        results = {
            f"MRR@{self.k}": float(np.mean(all_rr)),
            f"Recall@{self.k}": float(np.mean(all_hits)),
            "total_examples": len(all_rr),
            "segments": {}
        }

        for seg_name, data in segments.items():
            count = len(data["rr"])
            if count > 0:
                results["segments"][seg_name] = {
                    f"MRR@{self.k}": float(np.mean(data["rr"])),
                    f"Recall@{self.k}": float(np.mean(data["hits"])),
                    "count": count,
                    "pct_of_total": round(100.0 * count / len(all_rr), 2)
                }

        return results

    def print_report(self, results: Dict[str, Any], model_name: str = "Model"):
        print(f"\n=======================================================")
        print(f"  EVALUATION REPORT: {model_name} (Total: {results['total_examples']:,} examples)")
        print(f"=======================================================")
        print(f"  Overall MRR@{self.k}:    {results[f'MRR@{self.k}']:.4f}")
        print(f"  Overall Recall@{self.k}: {results[f'Recall@{self.k}']:.4f}")
        print(f"-------------------------------------------------------")
        print(f"  {'Segment':<19} | {'Count':<7} | {'% Total':<7} | {'MRR@' + str(self.k):<8} | {'Recall@' + str(self.k)}")
        print(f"-------------------------------------------------------")
        for seg, vals in results["segments"].items():
            print(f"  {seg:<19} | {vals['count']:<7} | {vals['pct_of_total']:<6}% | {vals[f'MRR@{self.k}']:.4f}   | {vals[f'Recall@{self.k}']:.4f}")
        print(f"=======================================================\n")
