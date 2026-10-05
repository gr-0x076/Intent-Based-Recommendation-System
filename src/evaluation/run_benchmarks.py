"""
Benchmark runner: Trains and evaluates models on the processed validation set.
Outputs structured metrics comparison table comparing:
- Level 0: Global Popularity
- Level 1: ItemCF (Co-occurrence + Recency Decay)
- Level 2: GRU4Rec (Sequential RNN)
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

from evaluation.metrics import EvaluationHarness


def run_benchmark(data_dir: str, k: int = 20, max_eval_samples: int = None, run_gru: bool = True):
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

    # 1. Level 0: Popularity
    print("\n[1/3] Evaluating Level 0: Global Popularity...")
    pop_model = PopularityRecommender(filter_seen=True)
    pop_model.fit(train_sessions)
    pop_res = harness.evaluate(pop_model, val_examples)
    harness.print_report(pop_res, "Level 0: Global Popularity")

    results_summary.append({
        "Model": "Level 0: Global Popularity",
        f"MRR@{k}": pop_res[f"MRR@{k}"],
        f"Recall@{k}": pop_res[f"Recall@{k}"],
        "Cold-1 MRR": pop_res["segments"].get("cold_start_1", {}).get(f"MRR@{k}", 0.0),
        "Cold-2 MRR": pop_res["segments"].get("cold_start_2", {}).get(f"MRR@{k}", 0.0),
        "Rich 3+ MRR": pop_res["segments"].get("rich_3plus", {}).get(f"MRR@{k}", 0.0),
        "Has-Search MRR": pop_res["segments"].get("has_search", {}).get(f"MRR@{k}", 0.0),
        "No-Search MRR": pop_res["segments"].get("no_search", {}).get(f"MRR@{k}", 0.0),
    })

    # 2. Level 1: ItemCF
    print("\n[2/3] Evaluating Level 1: ItemCF (Co-occurrence + Recency)...")
    itemcf_model = ItemCFRecommender(top_similar_items=100, recency_decay=0.7, filter_seen=True)
    itemcf_model.fit(train_sessions)
    itemcf_res = harness.evaluate(itemcf_model, val_examples)
    harness.print_report(itemcf_res, "Level 1: ItemCF (Recency=0.7)")

    results_summary.append({
        "Model": "Level 1: ItemCF",
        f"MRR@{k}": itemcf_res[f"MRR@{k}"],
        f"Recall@{k}": itemcf_res[f"Recall@{k}"],
        "Cold-1 MRR": itemcf_res["segments"].get("cold_start_1", {}).get(f"MRR@{k}", 0.0),
        "Cold-2 MRR": itemcf_res["segments"].get("cold_start_2", {}).get(f"MRR@{k}", 0.0),
        "Rich 3+ MRR": itemcf_res["segments"].get("rich_3plus", {}).get(f"MRR@{k}", 0.0),
        "Has-Search MRR": itemcf_res["segments"].get("has_search", {}).get(f"MRR@{k}", 0.0),
        "No-Search MRR": itemcf_res["segments"].get("no_search", {}).get(f"MRR@{k}", 0.0),
    })

    # 3. Level 2: GRU4Rec
    if run_gru and has_gru:
        print("\n[3/3] Evaluating Level 2: GRU4Rec (Sequential RNN)...")
        gru_model = GRU4RecRecommender(embed_dim=64, hidden_dim=64, lr=0.003, epochs=4, filter_seen=True)
        gru_model.fit(train_sessions)
        gru_res = harness.evaluate(gru_model, val_examples)
        harness.print_report(gru_res, "Level 2: GRU4Rec")

        results_summary.append({
            "Model": "Level 2: GRU4Rec",
            f"MRR@{k}": gru_res[f"MRR@{k}"],
            f"Recall@{k}": gru_res[f"Recall@{k}"],
            "Cold-1 MRR": gru_res["segments"].get("cold_start_1", {}).get(f"MRR@{k}", 0.0),
            "Cold-2 MRR": gru_res["segments"].get("cold_start_2", {}).get(f"MRR@{k}", 0.0),
            "Rich 3+ MRR": gru_res["segments"].get("rich_3plus", {}).get(f"MRR@{k}", 0.0),
            "Has-Search MRR": gru_res["segments"].get("has_search", {}).get(f"MRR@{k}", 0.0),
            "No-Search MRR": gru_res["segments"].get("no_search", {}).get(f"MRR@{k}", 0.0),
        })

    df = pd.DataFrame(results_summary)
    print("\n=============================================================================================================")
    print("                                      BENCHMARK COMPARISON TABLE")
    print("=============================================================================================================")
    print(df.to_string(index=False))
    print("=============================================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/mohith/intent-rec/data/processed/dev_3pct")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--skip-gru", action="store_true")
    args = parser.parse_args()

    run_benchmark(args.data_dir, k=args.k, max_eval_samples=args.max_eval, run_gru=not args.skip_gru)
