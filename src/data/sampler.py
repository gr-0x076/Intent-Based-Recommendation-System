"""
Streaming Session Sampler and Dataset Pipeline for Intent-Based Recommender.
Features:
1. Hash-based deterministic sampling (zero-memory session selection across browsing & search).
2. Chunk-based streaming read of raw CSVs to stay within safe RAM limits (<500MB).
3. Session aggregation with strictly chronological sorting by server_timestamp_epoch_ms.
4. Chronological Train (70%) / Validation (15%) / Test (15%) split by session start time.
5. Prefix-expansion example generator for evaluation (preserving target_item and target_action).
"""

import os
import csv
import ast
import json
import zlib
import argparse
from typing import Dict, List, Any, Tuple
import pandas as pd
from tqdm import tqdm


def is_session_sampled(session_id: str, sample_pct: float, seed: int = 42) -> bool:
    """Deterministic, zero-memory hash check to decide if a session is in the sample."""
    # Using adler32 with seed salt for speed and uniform distribution
    val = zlib.adler32(f"{session_id}_{seed}".encode("utf-8")) % 10000
    return val < int(sample_pct * 100)


def stream_browsing_events(
    browsing_path: str,
    sample_pct: float,
    chunk_size: int = 500_000,
    max_rows: int = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Streams browsing_train.csv in chunks and collects events for sampled sessions.
    Returns: dict of session_id -> list of browsing events
    """
    sessions = {}
    print(f"--> Streaming browsing events from {browsing_path} (sample={sample_pct}%)...")

    rows_read = 0
    with open(browsing_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Browsing events", unit="rows", mininterval=2.0):
            rows_read += 1
            if max_rows and rows_read > max_rows:
                break

            sid = row["session_id_hash"]
            if not is_session_sampled(sid, sample_pct):
                continue

            event = {
                "type": "product" if row["product_sku_hash"] else "pageview",
                "action": row["product_action"] or "pageview",
                "item_id": row["product_sku_hash"] or None,
                "url_hash": row["hashed_url"] or None,
                "timestamp_ms": int(row["server_timestamp_epoch_ms"]),
                "query_vector": None,
            }
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(event)

    print(f"    Loaded {sum(len(v) for v in sessions.values()):,} browsing events across {len(sessions):,} sessions.")
    return sessions


def stream_search_events(
    search_path: str,
    sample_pct: float,
    sessions: Dict[str, List[Dict[str, Any]]],
    max_rows: int = None
) -> None:
    """
    Streams search_train.csv and integrates search queries into existing sessions.
    Modifies sessions dict in-place.
    """
    if not os.path.exists(search_path):
        print(f"Search file {search_path} not found. Skipping search integration.")
        return

    print(f"--> Streaming search events from {search_path}...")
    rows_read = 0
    search_events_added = 0

    with open(search_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Search events", unit="rows", mininterval=2.0):
            rows_read += 1
            if max_rows and rows_read > max_rows:
                break

            sid = row["session_id_hash"]
            if not is_session_sampled(sid, sample_pct):
                continue

            qv_raw = row.get("query_vector")
            qv = ast.literal_eval(qv_raw) if qv_raw and qv_raw.startswith("[") else None

            clicked_raw = row.get("clicked_skus_hash")
            clicked = ast.literal_eval(clicked_raw) if clicked_raw and clicked_raw.startswith("[") else []

            event = {
                "type": "search",
                "action": "search",
                "item_id": None,
                "url_hash": None,
                "timestamp_ms": int(row["server_timestamp_epoch_ms"]),
                "query_vector": qv,
                "clicked_skus": clicked,
            }

            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(event)
            search_events_added += 1

    print(f"    Integrated {search_events_added:,} search events into sessions.")


def sort_and_filter_sessions(
    sessions: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    1. Sorts all events within each session chronologically by timestamp_ms.
    2. Drops sessions with 0 total events.
    """
    print("--> Chronologically sorting events within sessions...")
    cleaned = {}
    for sid, events in sessions.items():
        if not events:
            continue
        events.sort(key=lambda x: x["timestamp_ms"])
        cleaned[sid] = events
    return cleaned


def split_sessions_chronologically(
    sessions: Dict[str, List[Dict[str, Any]]],
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Splits sessions chronologically by the start timestamp of each session.
    70% Train, 15% Validation, 15% Test.
    """
    print("--> Performing chronological session split...")
    # Get session start time
    session_start_times = [(sid, events[0]["timestamp_ms"]) for sid, events in sessions.items()]
    session_start_times.sort(key=lambda x: x[1])

    n_total = len(session_start_times)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * (train_frac + val_frac))

    train_sids = {sid for sid, _ in session_start_times[:n_train]}
    val_sids = {sid for sid, _ in session_start_times[n_train:n_val]}
    test_sids = {sid for sid, _ in session_start_times[n_val:]}

    train_sessions = {sid: sessions[sid] for sid in train_sids}
    val_sessions = {sid: sessions[sid] for sid in val_sids}
    test_sessions = {sid: sessions[sid] for sid in test_sids}

    print(f"    Train sessions: {len(train_sessions):,} ({train_frac*100:.0f}%)")
    print(f"    Val sessions:   {len(val_sessions):,} ({val_frac*100:.0f}%)")
    print(f"    Test sessions:  {len(test_sessions):,} ({(1 - train_frac - val_frac)*100:.0f}%)")

    return train_sessions, val_sessions, test_sessions


def generate_prefix_examples(
    sessions: Dict[str, List[Dict[str, Any]]],
    min_prior_products: int = 1,
    max_examples_per_session: int = 10
) -> List[Dict[str, Any]]:
    """
    Generates prefix-expansion evaluation/training examples.
    For each session, every product interaction at step t (where prior product interactions >= min_prior_products)
    forms an example:
      - input_events: all events up to t-1
      - target_item_id: item at t
      - target_action: action at t (detail, add, purchase)
      - n_prior_products: count of product interactions in input_events
      - has_search: whether input_events contains any search query
    """
    examples = []
    for sid, events in sessions.items():
        # Identify indices where a product interaction occurs
        product_indices = [i for i, e in enumerate(events) if e["item_id"] is not None]

        if len(product_indices) < (min_prior_products + 1):
            continue

        session_ex_count = 0
        for k in range(min_prior_products, len(product_indices)):
            target_idx = product_indices[k]
            target_event = events[target_idx]

            input_events = events[:target_idx]

            # Count how many product interactions are in input_events
            n_priors = sum(1 for e in input_events if e["item_id"] is not None)
            has_search = any(e["type"] == "search" for e in input_events)

            examples.append({
                "session_id": sid,
                "input_events": input_events,
                "target_item_id": target_event["item_id"],
                "target_action": target_event["action"],
                "n_prior_products": n_priors,
                "has_search": has_search,
            })
            session_ex_count += 1
            if session_ex_count >= max_examples_per_session:
                break

    return examples


def save_processed_dataset(
    output_dir: str,
    train_sessions: Dict[str, Any],
    val_sessions: Dict[str, Any],
    val_examples: List[Dict[str, Any]],
    test_sessions: Dict[str, Any],
    test_examples: List[Dict[str, Any]]
) -> None:
    """Saves processed sessions and evaluation examples as parquet/pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    import pickle

    print(f"--> Saving processed datasets to {output_dir}...")
    with open(os.path.join(output_dir, "train_sessions.pkl"), "wb") as f:
        pickle.dump(train_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "val_sessions.pkl"), "wb") as f:
        pickle.dump(val_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "val_examples.pkl"), "wb") as f:
        pickle.dump(val_examples, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "test_sessions.pkl"), "wb") as f:
        pickle.dump(test_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "test_examples.pkl"), "wb") as f:
        pickle.dump(test_examples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save a lightweight manifest JSON
    manifest = {
        "train_sessions_count": len(train_sessions),
        "val_sessions_count": len(val_sessions),
        "val_examples_count": len(val_examples),
        "test_sessions_count": len(test_sessions),
        "test_examples_count": len(test_examples),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("--> Done. Manifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and process Coveo e-commerce session dataset")
    parser.add_argument("--browsing-path", type=str, default="/home/mohith/intent-rec/data/raw/browsing_train.csv")
    parser.add_argument("--search-path", type=str, default="/home/mohith/intent-rec/data/raw/search_train.csv")
    parser.add_argument("--output-dir", type=str, default="/home/mohith/intent-rec/data/processed")
    parser.add_argument("--sample-pct", type=float, default=2.0, help="Sampling percentage (e.g. 2.0 for 2%%)")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick testing")
    args = parser.parse_args()

    sessions = stream_browsing_events(args.browsing_path, sample_pct=args.sample_pct, max_rows=args.max_rows)
    stream_search_events(args.search_path, sample_pct=args.sample_pct, sessions=sessions, max_rows=args.max_rows)
    sessions = sort_and_filter_sessions(sessions)

    train_s, val_s, test_s = split_sessions_chronologically(sessions, train_frac=0.70, val_frac=0.15)

    print("--> Generating prefix evaluation examples for Validation set...")
    val_examples = generate_prefix_examples(val_s)
    print(f"    Generated {len(val_examples):,} validation examples.")

    print("--> Generating prefix evaluation examples for Test set...")
    test_examples = generate_prefix_examples(test_s)
    print(f"    Generated {len(test_examples):,} test examples.")

    save_processed_dataset(args.output_dir, train_s, val_s, val_examples, test_s, test_examples)
