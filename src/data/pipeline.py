"""
Experimental data pipeline.

Status:
Planned / under development.

Responsibilities:
- Load raw data
- Construct sessions
- Sort events temporally
- Create train/evaluation splits
- Prepare features for experiments
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List


class CoveoDataPipeline:
    """Loads and preprocesses the Coveo SIGIR 2021 dataset into session format."""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def load_raw(self, filename: str = "browsing_train.csv") -> pd.DataFrame:
        """
        Load raw Coveo CSV into a DataFrame.
        Returns: pd.DataFrame with raw event log
        """
        # TODO: read CSV from self.raw_dir / filename
        # TODO: validate required columns exist: session_id, server_timestamp_epoch_ms,
        #       event_type, product_sku_hash, query
        # TODO: drop rows where session_id is null
        # TODO: return dataframe
        pass

    def build_sessions(self, df: pd.DataFrame) -> Dict[str, List[dict]]:
        """
        Convert flat event log into structured sessions.
        Each session = list of events sorted by timestamp.
        Returns: dict {session_id: [events]}
        """
        # TODO: group df by session_id
        # TODO: sort each group by server_timestamp_epoch_ms
        # TODO: convert each row to event dict with keys: event_type, item_id, query, timestamp
        # TODO: return sessions dict
        pass

    def filter_sessions(
        self, sessions: Dict[str, List[dict]], min_length: int = 2
    ) -> Dict[str, List[dict]]:
        """
        Remove sessions that are too short to be useful.
        Returns: filtered sessions dict
        """
        # TODO: keep only sessions with len >= min_length
        # TODO: log how many sessions were filtered
        # TODO: return filtered dict
        pass

    def extract_query_sessions(
        self, sessions: Dict[str, List[dict]]
    ) -> Dict[str, List[dict]]:
        """
        Filter to sessions that contain at least one search query event.
        Returns: dict of sessions with at least one query event
        """
        # TODO: iterate sessions and keep only those with at least one event
        #       where event_type == 'query'
        # TODO: return filtered dict
        pass

    def save_processed(
        self, sessions: Dict[str, List[dict]], filename: str = "sessions.pkl"
    ) -> None:
        """
        Serialize processed sessions to disk.
        """
        # TODO: ensure self.processed_dir exists (mkdir parents=True)
        # TODO: pickle sessions to self.processed_dir / filename
        # TODO: print confirmation message with session count and output path
        pass

    def run(self) -> Dict[str, List[dict]]:
        """
        End-to-end pipeline: load → build sessions → filter → save.
        Returns: processed sessions dict
        """
        # TODO: call load_raw()
        # TODO: call build_sessions()
        # TODO: call filter_sessions()
        # TODO: call extract_query_sessions()
        # TODO: call save_processed()
        # TODO: return sessions
        pass
