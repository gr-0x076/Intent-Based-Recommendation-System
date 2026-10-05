"""
Intent Representation Probing Suite.
Evaluates whether learned representations (Model 2, Model 3A, Model 3B-λ0, Model 3B-λ0.1)
encode information related to operational intent:
1. Primary Probe: Behavioral Mode (Discovery [unseen item] vs. Conversion [repeat item])
   - Discovery: Target NOT present in input prefix
   - Conversion: Target present in input prefix
2. Secondary Probe: Target Action Type (detail vs. add vs. purchase)

Uses standard linear probing (LogisticRegression) on holdout split of validation representations.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.gru4rec import GRU4RecRecommender
from models.gru4rec_features import GRU4RecFeaturesRecommender
from models.gru4rec_intent import GRU4RecIntentRecommender


ACTION_LABEL_MAP = {
    "detail": 0,
    "add": 1,
    "purchase": 2
}


def run_probing_suite(data_dir: str, epochs: int = 4):
    print(f"--> [Probe] Loading dataset from {data_dir}...")
    with open(os.path.join(data_dir, "train_sessions.pkl"), "rb") as f:
        train_sessions = pickle.load(f)

    with open(os.path.join(data_dir, "val_examples.pkl"), "rb") as f:
        val_examples = pickle.load(f)

    print(f"    Train sessions: {len(train_sessions):,}, Validation examples: {len(val_examples):,}")

    # Build targets for all validation examples
    y_mode = []       # 0: Discovery (unseen), 1: Conversion (repeat)
    y_action = []     # 0: detail, 1: add, 2: purchase
    valid_mask = []

    for ex in val_examples:
        tgt_item = ex.get("target_item_id") or ex.get("target_item")
        prefix_items = set(e["item_id"] for e in ex["input_events"] if e.get("item_id"))
        is_repeat = 1 if tgt_item in prefix_items else 0
        y_mode.append(is_repeat)

        act = ex.get("target_action", "detail")
        act_idx = ACTION_LABEL_MAP.get(act, 0)
        y_action.append(act_idx)

    y_mode = np.array(y_mode, dtype=np.int32)
    y_action = np.array(y_action, dtype=np.int32)

    print(f"    Labels generated: Mode Conversion={y_mode.mean()*100:.1f}%, Action breakdown={np.bincount(y_action)}")

    # 1. Random Baseline Representation
    np.random.seed(42)
    rep_random = np.random.randn(len(val_examples), 64).astype(np.float32)

    models_to_probe = {}

    # Train Models
    print("\n--- Training Model 2: Pure GRU ---")
    m2 = GRU4RecRecommender(embed_dim=64, hidden_dim=64, lr=0.003, epochs=epochs, filter_seen=True)
    m2.fit(train_sessions)
    models_to_probe["Model 2 (Pure GRU, 64-D)"] = m2

    print("\n--- Training Model 3A: GRU + Raw Features (Control) ---")
    m3a = GRU4RecFeaturesRecommender(epochs=epochs, filter_seen=True)
    m3a.fit(train_sessions)
    models_to_probe["Model 3A (Raw Features, 112-D)"] = m3a

    print("\n--- Training Model 3B-λ0: Pure Latent Bottleneck ---")
    m3b_l0 = GRU4RecIntentRecommender(lambda_aux=0.0, epochs=epochs, filter_seen=True)
    m3b_l0.fit(train_sessions)
    models_to_probe["Model 3B-λ0 (Pure Bottleneck, 64-D)"] = m3b_l0

    print("\n--- Training Model 3B-λ0.1: Supervised Bottleneck ---")
    m3b_l01 = GRU4RecIntentRecommender(lambda_aux=0.1, epochs=epochs, filter_seen=True)
    m3b_l01.fit(train_sessions)
    models_to_probe["Model 3B-λ0.1 (Supervised, 64-D)"] = m3b_l01

    # Extract Representations
    all_reps = {"Random Control (64-D)": rep_random}
    for name, model in models_to_probe.items():
        print(f"--> Extracting representations for {name}...")
        reps = []
        for ex in tqdm(val_examples, desc=name):
            r = model.extract_representation(ex["input_events"])
            reps.append(r)
        all_reps[name] = np.array(reps, dtype=np.float32)

    # Split indices (80% train probe, 20% test probe)
    idx_train, idx_test = train_test_split(np.arange(len(val_examples)), test_size=0.20, random_state=42, stratify=y_mode)

    print("\n=========================================================================================")
    print("                      INTENT PROBING RESULTS (Holdout N = 4,994)")
    print("=========================================================================================")
    print(f"{'Representation':<36} | {'Mode Acc':<9} | {'Mode AUC':<9} | {'Action Acc':<11} | {'Action F1':<9}")
    print("-" * 89)

    for name, X in all_reps.items():
        X_tr, X_te = X[idx_train], X[idx_test]

        # Probe 1: Discovery vs. Conversion Mode
        clf_mode = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf_mode.fit(X_tr, y_mode[idx_train])
        preds_mode = clf_mode.predict(X_te)
        probs_mode = clf_mode.predict_proba(X_te)[:, 1]

        acc_mode = accuracy_score(y_mode[idx_test], preds_mode)
        auc_mode = roc_auc_score(y_mode[idx_test], probs_mode)

        # Probe 2: Target Action Type
        clf_act = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf_act.fit(X_tr, y_action[idx_train])
        preds_act = clf_act.predict(X_te)

        acc_act = accuracy_score(y_action[idx_test], preds_act)
        f1_act = f1_score(y_action[idx_test], preds_act, average="macro")

        print(f"{name:<36} | {acc_mode*100:6.2f}%   | {auc_mode:7.4f}   | {acc_act*100:8.2f}%   | {f1_act:7.4f}")

    print("=========================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/mohith/intent-rec/data/processed/dev_3pct")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    run_probing_suite(args.data_dir, epochs=args.epochs)
