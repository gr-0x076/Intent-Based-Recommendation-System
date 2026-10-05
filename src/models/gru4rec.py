"""
Level 2 Sequential Baseline: Clean Canonical GRU4Rec.
Architecture:
- Item Embedding layer: Maps product item IDs to dense d-dimensional vectors.
- Single-layer GRU: Encodes temporal transition dynamics of product views.
- Scoring Head: Projects session hidden state to vocabulary logits (CrossEntropyLoss).
- STRICT ABLATION RULE: Pure item sequences only. No search vectors, no action weights, no attention.
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None


class SessionSeqDataset(Dataset):
    """Dataset of (sequence of item IDs, target item ID) pairs."""
    def __init__(self, sequences: List[List[int]], targets: List[int], max_len: int = 20):
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        pad_len = self.max_len - len(seq)
        padded_seq = [0] * pad_len + seq
        return (
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class GRU4RecNet(nn.Module if nn else object):
    def __init__(self, num_items: int, embed_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, num_items, bias=True)

    def forward(self, x):
        # x: [batch_size, max_len]
        embeds = self.dropout(self.item_embedding(x))  # [batch_size, max_len, embed_dim]
        out, h_n = self.gru(embeds)                    # h_n: [1, batch_size, hidden_dim]
        session_repr = h_n.squeeze(0)                  # [batch_size, hidden_dim]
        logits = self.fc(session_repr)                 # [batch_size, num_items]
        return logits


class GRU4RecRecommender:
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 0.003,
        batch_size: int = 512,
        epochs: int = 5,
        min_freq: int = 3,
        filter_seen: bool = True,
        device: str = "cpu"
    ):
        self.embed_dim = embed_dim
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
        print(f"--> [GRU4Rec] Vocabulary: {len(self.item2idx):,} items (min_freq={self.min_freq})")

    def prepare_training_pairs(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        sequences = []
        targets = []
        unk_idx = self.item2idx["<UNK>"]

        for sid, events in train_sessions.items():
            # Get clean chronological product interaction sequence
            prod_seq = [self.item2idx.get(e["item_id"], unk_idx) for e in events if e.get("item_id")]
            if len(prod_seq) < 2:
                continue

            for t in range(1, len(prod_seq)):
                inp = prod_seq[:t]
                tgt = prod_seq[t]
                if tgt != unk_idx:  # do not train on UNK target
                    sequences.append(inp)
                    targets.append(tgt)

        print(f"--> [GRU4Rec] Generated {len(targets):,} training sequence pairs.")
        return sequences, targets

    def fit(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        if torch is None:
            raise ImportError("PyTorch is required for GRU4Rec. Please install torch.")

        self.build_vocab(train_sessions)
        sequences, targets = self.prepare_training_pairs(train_sessions)

        dataset = SessionSeqDataset(sequences, targets, max_len=20)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = GRU4RecNet(
            num_items=len(self.item2idx),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        print(f"--> [GRU4Rec] Training for {self.epochs} epochs on {self.device}...")
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            steps = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(1, steps)
            print(f"    Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(self, input_events: List[Dict[str, Any]], top_k: int = 20) -> List[str]:
        if self.model is None or torch is None:
            return self.popular_items[:top_k]

        unk_idx = self.item2idx.get("<UNK>", 1)
        seen_items = set()
        prod_indices = []

        for e in input_events:
            item = e.get("item_id")
            if item:
                seen_items.add(item)
                prod_indices.append(self.item2idx.get(item, unk_idx))

        if not prod_indices:
            # Cold-start fallback to popularity
            return [it for it in self.popular_items if it not in seen_items][:top_k]

        # Truncate and pad sequence to max_len 20
        seq = prod_indices[-20:]
        padded = [0] * (20 - len(seq)) + seq
        tensor_x = torch.tensor([padded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(tensor_x).squeeze(0)  # [num_items]

            # Mask out PAD, UNK
            logits[0] = -1e9
            logits[1] = -1e9

            # Mask out seen items if filter_seen is enabled
            if self.filter_seen:
                for item in seen_items:
                    idx = self.item2idx.get(item)
                    if idx and idx < len(logits):
                        logits[idx] = -1e9

            # Extract top_k items
            top_indices = torch.topk(logits, k=top_k).indices.cpu().numpy()

        recommendations = [self.idx2item[idx] for idx in top_indices if idx in self.idx2item]

        # Fill remaining slots with popularity if needed
        if len(recommendations) < top_k:
            for it in self.popular_items:
                if it not in seen_items and it not in recommendations:
                    recommendations.append(it)
                if len(recommendations) >= top_k:
                    break

        return recommendations[:top_k]

    def extract_representation(self, input_events: List[Dict[str, Any]]) -> np.ndarray:
        if self.model is None or torch is None:
            return np.zeros(self.hidden_dim, dtype=np.float32)

        unk_idx = self.item2idx.get("<UNK>", 1)
        prod_indices = [self.item2idx.get(e["item_id"], unk_idx) for e in input_events if e.get("item_id")]
        if not prod_indices:
            return np.zeros(self.hidden_dim, dtype=np.float32)

        seq = prod_indices[-20:]
        padded = [0] * (20 - len(seq)) + seq
        tensor_x = torch.tensor([padded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            embeds = self.model.item_embedding(tensor_x)
            _, h_n = self.model.gru(embeds)
            return h_n.squeeze(0).squeeze(0).cpu().numpy()
