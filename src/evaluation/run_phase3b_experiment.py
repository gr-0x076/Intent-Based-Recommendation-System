"""
Dedicated Runner for Phase 3B Experiments:
- 3B-λ0: Pure Latent Bottleneck (λ=0.0) [Primary Experiment]
- 3B-λ0.1: Supervised Intent Bottleneck (λ=0.1) [Ablation Experiment]

Evaluates on the exact frozen validation set (N = 24,968) using EvaluationHarness.
"""

import os
import sys
import pickle
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.gru4rec_intent import GRU4RecIntentRecommender
from evaluation.metrics import EvaluationHarness


def run_phase3b(data_dir: str, k: int = 20):
    print(f"--> [Phase 3B] Loading dataset from {data_dir}...")
    with open(os.path.join(data_dir, "train_sessions.pkl"), "rb") as f:
        train_sessions = pickle.load(f)

    with open(os.path.join(data_dir, "val_examples.pkl"), "rb") as f:
        val_examples = pickle.load(f)

    print(f"    Train sessions: {len(train_sessions):,}, Validation examples: {len(val_examples):,}")

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

    # Benchmark References for Context
    results_summary.append({
        "Model": "Level 1: ItemCF (Ref)",
        f"MRR@{k}": 0.1440,
        f"Recall@{k}": 0.2767,
        "Cold-1 MRR": 0.1875,
        "Cold-2 MRR": 0.1541,
        "Rich 3+ MRR": 0.1171,
        "Has-Search MRR": 0.1100,
        "No-Search MRR": 0.1480,
        "Discovery MRR": 0.2489,
        "Conversion MRR": 0.0000,
    })
    results_summary.append({
        "Model": "Level 2: Pure GRU (Ref)",
        f"MRR@{k}": 0.1161,
        f"Recall@{k}": 0.2011,
        "Cold-1 MRR": 0.1525,
        "Cold-2 MRR": 0.1249,
        "Rich 3+ MRR": 0.0934,
        "Has-Search MRR": 0.0835,
        "No-Search MRR": 0.1199,
        "Discovery MRR": 0.1944,
        "Conversion MRR": 0.0000,
    })
    results_summary.append({
        "Model": "Level 3A: GRU+Features (Control)",
        f"MRR@{k}": 0.1216,
        f"Recall@{k}": 0.2130,
        "Cold-1 MRR": 0.1609,
        "Cold-2 MRR": 0.1300,
        "Rich 3+ MRR": 0.0974,
        "Has-Search MRR": 0.0896,
        "No-Search MRR": 0.1253,
        "Discovery MRR": 0.2035,
        "Conversion MRR": 0.0000,
    })

    # 1. Level 3B-λ0: Pure Latent Bottleneck
    print("\n=======================================================")
    print(" [1/2] Training & Evaluating Level 3B-λ0 (Pure Bottleneck)")
    print("=======================================================")
    m3b_l0 = GRU4RecIntentRecommender(lambda_aux=0.0, epochs=4, filter_seen=True)
    m3b_l0.fit(train_sessions)
    res_l0 = harness.evaluate(m3b_l0, val_examples)
    harness.print_report(res_l0, "Level 3B-λ0: Pure Latent Bottleneck")
    record_result("Level 3B-λ0: Bottleneck", res_l0)

    # 2. Level 3B-λ0.1: Supervised Intent Bottleneck
    print("\n=======================================================")
    print(" [2/2] Training & Evaluating Level 3B-λ0.1 (Supervised Bottleneck)")
    print("=======================================================")
    m3b_l01 = GRU4RecIntentRecommender(lambda_aux=0.1, epochs=4, filter_seen=True)
    m3b_l01.fit(train_sessions)
    res_l01 = harness.evaluate(m3b_l01, val_examples)
    harness.print_report(res_l01, "Level 3B-λ0.1: Supervised Bottleneck")
    record_result("Level 3B-λ0.1: Supervised", res_l01)

    df = pd.DataFrame(results_summary)
    print("\n" + "=" * 125)
    print("                                      PHASE 3B EVALUATION RESULTS (N = 24,968)")
    print("=" * 125)
    print(df.to_string(index=False))
    print("=" * 125 + "\n")

    # Save results to pickle
    out_path = os.path.join(data_dir, "phase3b_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"l0": res_l0, "l01": res_l01, "df": df}, f)
    print(f"--> Saved Phase 3B benchmark results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/mohith/intent-rec/data/processed/dev_3pct")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    run_phase3b(args.data_dir, k=args.k)
