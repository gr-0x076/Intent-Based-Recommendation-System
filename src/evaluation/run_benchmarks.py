"""
Automated Multi-Model Benchmark Runner.
Evaluates:
- Level 0: Global Popularity
- Level 1: ItemCF (Co-occurrence + Recency Decay)
- Level 2: Pure GRU4Rec (Item Sequences Only)
- Level 3A: GRU4Rec + Raw Features (Concatenated Action + Query) [Phase 3 Control]
- Sanity Check: Repeat Last Item Prior
"""

import os
import sys
import pickle
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.popularity import PopularityRecommender
from models.itemcf import ItemCFRecommender
try:
    from models.gru4rec import GRU4RecRecommender
    has_gru = True
except ImportError:
    has_gru = False

try:
    from models.gru4rec_features import GRU4RecFeaturesRecommender
    has_gru_feat = True
except ImportError:
    has_gru_feat = False

from evaluation.metrics import EvaluationHarness


class RepeatLastItemRecommender:
    def predict(self, input_events, top_k=20):
        prods = [e["item_id"] for e in input_events if e.get("item_id")]
        return [prods[-1]] if prods else []


def run_benchmark(data_dir: str, k: int = 20, max_eval_samples: int = None, run_nn: bool = True):
    print(f"--> Loading dataset from {data_dir}...")
    with open(os.path.join(data_dir, "train_sessions.pkl"), "rb") as f:
        train_sessions = pickle.load(f)

    with open(os.path.join(data_dir, "val_examples.pkl"), "rb") as f:
        val_examples = pickle.load(f)

    if max_eval_samples and len(val_examples) > max_eval_samples:
        print(f"    Subsampling {max_eval_samples:,} validation examples for fast evaluation.")
        val_examples = val_examples[:max_eval_samples]

    print(f"    Loaded {len(train_sessions):,} train sessions and {len(val_examples):,} validation examples.")

    harness = EvaluationHarness(k=k)
    results_summary = []

    def record_result(name, res):
        results_summary.append({
            "Model": name,
            f"MRR@{k}": res[f"MRR@{k}"],
            f"Recall@{k}": res[f"Recall@{k}"],
            "Cold-1 MRR": res["segments"].get("cold_start_1", {}).get(f"MRR@{k}", 0.0),
            "Cold-2 MRR": res["segments"].get("cold_start_2", {}).get(f"MRR@{k}", 0.0),
            "Rich 3+ MRR": res["segments"].get("rich_3plus", {}).get(f"MRR@{k}", 0.0),
            "Has-Search MRR": res["segments"].get("has_search", {}).get(f"MRR@{k}", 0.0),
            "No-Search MRR": res["segments"].get("no_search", {}).get(f"MRR@{k}", 0.0),
            "Discovery MRR": res["segments"].get("discovery_unseen", {}).get(f"MRR@{k}", 0.0),
            "Conversion MRR": res["segments"].get("conversion_repeat", {}).get(f"MRR@{k}", 0.0),
        })

    # 1. Level 0: Popularity
    print("\n[1/5] Evaluating Level 0: Global Popularity...")
    pop_model = PopularityRecommender(filter_seen=True)
    pop_model.fit(train_sessions)
    pop_res = harness.evaluate(pop_model, val_examples)
    harness.print_report(pop_res, "Level 0: Global Popularity")
    record_result("Level 0: Popularity", pop_res)

    # 2. Level 1: ItemCF
    print("\n[2/5] Evaluating Level 1: ItemCF (Co-occurrence + Recency)...")
    itemcf_model = ItemCFRecommender(top_similar_items=100, recency_decay=0.7, filter_seen=True)
    itemcf_model.fit(train_sessions)
    itemcf_res = harness.evaluate(itemcf_model, val_examples)
    harness.print_report(itemcf_res, "Level 1: ItemCF (Recency=0.7)")
    record_result("Level 1: ItemCF", itemcf_res)

    # 3. Sanity: Repeat Last Item
    print("\n[3/5] Evaluating Sanity Check: Repeat Last Item...")
    rep_model = RepeatLastItemRecommender()
    rep_res = harness.evaluate(rep_model, val_examples)
    harness.print_report(rep_res, "[Sanity] Repeat Last Item")
    record_result("[Sanity] Repeat Last Item", rep_res)

    # 4. Level 2: Pure GRU4Rec
    if run_nn and has_gru:
        print("\n[4/5] Evaluating Level 2: Pure GRU4Rec (Item Sequences Only)...")
        gru_model = GRU4RecRecommender(embed_dim=64, hidden_dim=64, lr=0.003, epochs=4, filter_seen=True)
        gru_model.fit(train_sessions)
        gru_res = harness.evaluate(gru_model, val_examples)
        harness.print_report(gru_res, "Level 2: Pure GRU4Rec")
        record_result("Level 2: Pure GRU4Rec", gru_res)

    # 5. Level 3A: GRU4Rec + Raw Features (Control)
    if run_nn and has_gru_feat:
        print("\n[5/5] Evaluating Level 3A: GRU4Rec + Raw Features (Control)...")
        gru_feat_model = GRU4RecFeaturesRecommender(epochs=4, filter_seen=True)
        gru_feat_model.fit(train_sessions)
        gru_feat_res = harness.evaluate(gru_feat_model, val_examples)
        harness.print_report(gru_feat_res, "Level 3A: GRU + Raw Features (Control)")
        record_result("Level 3A: GRU+Features", gru_feat_res)

    df = pd.DataFrame(results_summary)
    print("\n=========================================================================================================================")
    print("                                            BENCHMARK COMPARISON TABLE")
    print("=========================================================================================================================")
    print(df.to_string(index=False))
    print("=========================================================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/mohith/intent-rec/data/processed/dev_3pct")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--skip-nn", action="store_true")
    args = parser.parse_args()

    run_benchmark(args.data_dir, k=args.k, max_eval_samples=args.max_eval, run_nn=not args.skip_nn)
