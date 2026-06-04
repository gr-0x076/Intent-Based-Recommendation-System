from __future__ import annotations
import math
from typing import List


def recall_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """
    Compute Recall@K averaged over all sessions.

    predictions: list of ranked item id lists (one per session)
    ground_truth: list of correct next item ids (one per session)
    k: rank cutoff
    returns: float in [0, 1]
    """
    if not ground_truth:
        return 0.0
    hits = sum(1 for pred, gt in zip(predictions, ground_truth) if gt in pred[:k])
    return hits / len(ground_truth)


def mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank.

    predictions: list of ranked item id lists (one per session)
    ground_truth: list of correct next item ids (one per session)
    returns: float in [0, 1]
    """
    if not ground_truth:
        return 0.0
    rr_scores = []
    for pred, gt in zip(predictions, ground_truth):
        try:
            rank = pred.index(gt) + 1
            rr_scores.append(1.0 / rank)
        except ValueError:
            rr_scores.append(0.0)
    return sum(rr_scores) / len(rr_scores)


def ndcg_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """
    Compute NDCG@K averaged over all sessions.
    Assumes single relevant item per session (binary relevance).
    IDCG = 1/log2(2) = 1.0 (ideal: relevant item at rank 1).

    predictions: list of ranked item id lists (one per session)
    ground_truth: list of correct next item ids (one per session)
    k: rank cutoff
    returns: float in [0, 1]
    """
    if not ground_truth:
        return 0.0
    scores = []
    for pred, gt in zip(predictions, ground_truth):
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            if item == gt:
                dcg = 1.0 / math.log2(i + 2)
                break
        scores.append(dcg)
    return sum(scores) / len(scores)


def evaluate_all(predictions: List[List[str]], ground_truth: List[str]) -> dict:
    """
    Run the full standard evaluation suite.

    returns: dict with keys recall@5, recall@10, recall@20, mrr, ndcg@10
    """
    return {
        "recall@5": recall_at_k(predictions, ground_truth, 5),
        "recall@10": recall_at_k(predictions, ground_truth, 10),
        "recall@20": recall_at_k(predictions, ground_truth, 20),
        "mrr": mrr(predictions, ground_truth),
        "ndcg@10": ndcg_at_k(predictions, ground_truth, 10),
    }


def evaluate_intent_shift_sessions(
    predictions: List[List[str]],
    ground_truth: List[str],
    is_shift_session: List[bool],
) -> dict:
    """
    Separately evaluate intent-shift sessions vs normal sessions.

    is_shift_session: parallel boolean list — True if the session contains an intent shift
    returns: dict with keys 'intent_shift' and 'normal', each containing evaluate_all output
    """
    shift_preds, shift_gt = [], []
    normal_preds, normal_gt = [], []

    for pred, gt, is_shift in zip(predictions, ground_truth, is_shift_session):
        if is_shift:
            shift_preds.append(pred)
            shift_gt.append(gt)
        else:
            normal_preds.append(pred)
            normal_gt.append(gt)

    return {
        "intent_shift": evaluate_all(shift_preds, shift_gt) if shift_gt else {},
        "normal": evaluate_all(normal_preds, normal_gt) if normal_gt else {},
    }


if __name__ == "__main__":
    # fmt: off
    predictions = [
        ["item_3", "item_1", "item_5", "item_2", "item_4", "item_7", "item_8", "item_9", "item_10", "item_11",
         "item_12", "item_13", "item_14", "item_15", "item_16", "item_17", "item_18", "item_19", "item_20", "item_21"],
        ["item_1", "item_2", "item_3", "item_4", "item_5", "item_6", "item_7", "item_8", "item_9", "item_10",
         "item_11", "item_12", "item_13", "item_14", "item_15", "item_16", "item_17", "item_18", "item_19", "item_20"],
        ["item_9", "item_8", "item_7", "item_6", "item_5", "item_4", "item_3", "item_2", "item_1", "item_0",
         "item_10", "item_11", "item_12", "item_13", "item_14", "item_15", "item_16", "item_17", "item_18", "item_19"],
        ["item_0", "item_1", "item_2", "item_3", "item_4", "item_5", "item_6", "item_7", "item_8", "item_9",
         "item_10", "item_11", "item_12", "item_13", "item_14", "item_15", "item_16", "item_17", "item_18", "item_19"],
        ["item_5", "item_6", "item_7", "item_8", "item_9", "item_10", "item_11", "item_12", "item_13", "item_14",
         "item_15", "item_16", "item_17", "item_18", "item_19", "item_20", "item_21", "item_22", "item_23", "item_24"],
    ]
    # fmt: on
    ground_truth = ["item_1", "item_5", "item_7", "item_0", "item_99"]
    is_shift = [True, False, True, False, True]

    print("=== evaluate_all ===")
    results = evaluate_all(predictions, ground_truth)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    print("\n=== evaluate_intent_shift_sessions ===")
    shift_results = evaluate_intent_shift_sessions(predictions, ground_truth, is_shift)
    for split, metrics in shift_results.items():
        print(f"\n  [{split}]")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
