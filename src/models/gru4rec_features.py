"""
Level 3A Control Model: GRU with Concatenated Raw Features.
Purpose in Phase 3:
- Tests whether adding raw auxiliary features (action type, search query vector)
  improves recommendation performance via standard concatenation.
- Acts as the indispensable CONTROL model against which Model 3B (Latent Intent Bottleneck)
  must be compared.

Input per step t:
- Item embedding: 64-D
- Action embedding: 16-D (pageview, detail, add, remove, purchase, search)
- Query projection: 50-D pre-trained query vector projected to 32-D (or zero if no query)
Total step input dimension: 64 + 16 + 32 = 112-D
Sequence model: 1-layer GRU (hidden_dim=112)
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


ACTION_MAP = {
    "<PAD>": 0,
    "pageview": 1,
    "detail": 2,
    "add": 3,
    "remove": 4,
    "purchase": 5,
    "search": 6
}


class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        item_seqs: List[List[int]],
        action_seqs: List[List[int]],
        query_seqs: List[List[np.ndarray]],
        targets: List[int],
        max_len: int = 20
    ):
        self.item_seqs = item_seqs
        self.action_seqs = action_seqs
        self.query_seqs = query_seqs
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        items = self.item_seqs[idx][-self.max_len:]
        actions = self.action_seqs[idx][-self.max_len:]
        queries = self.query_seqs[idx][-self.max_len:]

        pad_len = self.max_len - len(items)

        padded_items = [0] * pad_len + items
        padded_actions = [0] * pad_len + actions

        zero_q = np.zeros(50, dtype=np.float32)
        padded_queries = [zero_q] * pad_len + queries
        padded_queries = np.array(padded_queries, dtype=np.float32)

        return (
            torch.tensor(padded_items, dtype=torch.long),
            torch.tensor(padded_actions, dtype=torch.long),
            torch.tensor(padded_queries, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class FeatureGRU4RecNet(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_dim: int = 64,
        action_dim: int = 16,
        query_dim: int = 50,
        query_proj_dim: int = 32,
        hidden_dim: int = 112,
        dropout: float = 0.2
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(len(ACTION_MAP) + 1, action_dim, padding_idx=0)
        self.query_proj = nn.Linear(query_dim, query_proj_dim)

        input_dim = item_dim + action_dim + query_proj_dim
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, num_items, bias=True)

    def forward(self, items, actions, queries):
        # items: [batch, max_len], actions: [batch, max_len], queries: [batch, max_len, 50]
        i_emb = self.item_embedding(items)       # [batch, max_len, 64]
        a_emb = self.action_embedding(actions)   # [batch, max_len, 16]
        q_emb = torch.relu(self.query_proj(queries)) # [batch, max_len, 32]

        x = torch.cat([i_emb, a_emb, q_emb], dim=-1) # [batch, max_len, 112]
        x = self.dropout(x)

        out, h_n = self.gru(x)
        session_repr = h_n.squeeze(0)                 # [batch, hidden_dim]
        logits = self.fc(session_repr)                # [batch, num_items]
        return logits


class GRU4RecFeaturesRecommender:
    def __init__(
        self,
        item_dim: int = 64,
        action_dim: int = 16,
        query_proj_dim: int = 32,
        hidden_dim: int = 112,
        lr: float = 0.003,
        batch_size: int = 512,
        epochs: int = 4,
        min_freq: int = 3,
        filter_seen: bool = True,
        device: str = "cpu"
    ):
        self.item_dim = item_dim
        self.action_dim = action_dim
        self.query_proj_dim = query_proj_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_freq = min_freq
        self.filter_seen = filter_seen
        self.device = device

        self.item2idx = {}
        self.idx2item = {}
        self.popular_items = []
        self.model = None

    def build_vocab(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        counts = Counter()
        for sid, events in train_sessions.items():
            for e in events:
                item = e.get("item_id")
                if item:
                    counts[item] += 1

        self.item2idx = {"<PAD>": 0, "<UNK>": 1}
        for item, count in counts.items():
            if count >= self.min_freq:
                self.item2idx[item] = len(self.item2idx)

        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.popular_items = [item for item, _ in counts.most_common()]
        print(f"--> [Model 3A: GRU+Features] Vocabulary: {len(self.item2idx):,} items (min_freq={self.min_freq})")

    def prepare_training_pairs(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        item_seqs = []
        action_seqs = []
        query_seqs = []
        targets = []
        unk_idx = self.item2idx["<UNK>"]

        for sid, events in train_sessions.items():
            # Collect full chronological event stream
            seq_items = []
            seq_actions = []
            seq_queries = []
            active_query = np.zeros(50, dtype=np.float32)

            product_indices = []
            for idx, e in enumerate(events):
                # Update running active query vector if search event occurred
                if e.get("type") == "search" and e.get("query_vector") is not None:
                    qv = e["query_vector"]
                    if len(qv) == 50:
                        active_query = np.array(qv, dtype=np.float32)

                item = e.get("item_id")
                item_idx = self.item2idx.get(item, unk_idx) if item else 0
                action = e.get("action", "pageview")
                act_idx = ACTION_MAP.get(action, 1)

                seq_items.append(item_idx)
                seq_actions.append(act_idx)
                seq_queries.append(active_query.copy())

                if item:
                    product_indices.append(idx)

            if len(product_indices) < 2:
                continue

            for k in range(1, len(product_indices)):
                tgt_idx = product_indices[k]
                tgt_item = seq_items[tgt_idx]

                if tgt_item != unk_idx and tgt_item != 0:
                    item_seqs.append(seq_items[:tgt_idx])
                    action_seqs.append(seq_actions[:tgt_idx])
                    query_seqs.append(seq_queries[:tgt_idx])
                    targets.append(tgt_item)

        print(f"--> [Model 3A: GRU+Features] Generated {len(targets):,} training examples.")
        return item_seqs, action_seqs, query_seqs, targets

    def fit(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        self.build_vocab(train_sessions)
        item_seqs, action_seqs, query_seqs, targets = self.prepare_training_pairs(train_sessions)

        dataset = FeatureSequenceDataset(item_seqs, action_seqs, query_seqs, targets, max_len=20)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = FeatureGRU4RecNet(
            num_items=len(self.item2idx),
            item_dim=self.item_dim,
            action_dim=self.action_dim,
            query_dim=50,
            query_proj_dim=self.query_proj_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        print(f"--> [Model 3A: GRU+Features] Training for {self.epochs} epochs on {self.device}...")
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            steps = 0
            for b_items, b_actions, b_queries, b_targets in loader:
                b_items = b_items.to(self.device)
                b_actions = b_actions.to(self.device)
                b_queries = b_queries.to(self.device)
                b_targets = b_targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(b_items, b_actions, b_queries)
                loss = criterion(logits, b_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(1, steps)
            print(f"    Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(self, input_events: List[Dict[str, Any]], top_k: int = 20) -> List[str]:
        if self.model is None:
            return self.popular_items[:top_k]

        unk_idx = self.item2idx.get("<UNK>", 1)
        seen_items = set()
        seq_items = []
        seq_actions = []
        seq_queries = []
        active_query = np.zeros(50, dtype=np.float32)

        for e in input_events:
            if e.get("type") == "search" and e.get("query_vector") is not None:
                qv = e["query_vector"]
                if len(qv) == 50:
                    active_query = np.array(qv, dtype=np.float32)

            item = e.get("item_id")
            if item:
                seen_items.add(item)
            item_idx = self.item2idx.get(item, unk_idx) if item else 0
            action = e.get("action", "pageview")
            act_idx = ACTION_MAP.get(action, 1)

            seq_items.append(item_idx)
            seq_actions.append(act_idx)
            seq_queries.append(active_query.copy())

        if not seq_items:
            return [it for it in self.popular_items if it not in seen_items][:top_k]

        items_20 = seq_items[-20:]
        actions_20 = seq_actions[-20:]
        queries_20 = seq_queries[-20:]

        pad_len = 20 - len(items_20)
        padded_items = [0] * pad_len + items_20
        padded_actions = [0] * pad_len + actions_20
        zero_q = np.zeros(50, dtype=np.float32)
        padded_queries = [zero_q] * pad_len + queries_20
        padded_queries = np.array([padded_queries], dtype=np.float32)

        t_items = torch.tensor([padded_items], dtype=torch.long, device=self.device)
        t_actions = torch.tensor([padded_actions], dtype=torch.long, device=self.device)
        t_queries = torch.tensor(padded_queries, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(t_items, t_actions, t_queries).squeeze(0)
            logits[0] = -1e9
            logits[1] = -1e9

            if self.filter_seen:
                for it in seen_items:
                    idx = self.item2idx.get(it)
                    if idx and idx < len(logits):
                        logits[idx] = -1e9

            top_indices = torch.topk(logits, k=top_k).indices.cpu().numpy()

        recommendations = [self.idx2item[idx] for idx in top_indices if idx in self.idx2item]

        if len(recommendations) < top_k:
            for it in self.popular_items:
                if (not self.filter_seen or it not in seen_items) and it not in recommendations:
                    recommendations.append(it)
                if len(recommendations) >= top_k:
                    break

        return recommendations[:top_k]
