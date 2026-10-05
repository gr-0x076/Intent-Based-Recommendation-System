"""
Level 3B: GRU with Latent Intent Bottleneck.
Purpose in Phase 3:
- Tests whether explicitly abstracting user intent into a latent informational bottleneck (z_intent)
  provides additional measurable value over simple raw-feature concatenation (Model 3A).
- Operates under two modes:
  * 3B-λ0: Pure latent bottleneck (λ=0.0, primary experiment)
  * 3B-λ0.1: Bottleneck with auxiliary intent supervision (λ=0.1, ablation experiment)

Architecture:
1. Sequential Context Encoder: GRU(items, actions, search query) -> hidden state h_t (112-D)
2. Latent Intent Bottleneck:
   z_intent = LayerNorm(W2 * GELU(W1 * [h_t; a_t; q_t] + b1) + b2) in R^64
3. Candidate Item Scoring:
   logits = W_catalog * z_intent (in shared learned scoring space)
4. Auxiliary Intent Head (for λ > 0):
   logits_mode = W_aux * z_intent (predicting Discovery vs. Conversion mode)
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
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


class IntentSequenceDataset(Dataset):
    def __init__(
        self,
        item_seqs: List[List[int]],
        action_seqs: List[List[int]],
        query_seqs: List[List[np.ndarray]],
        targets: List[int],
        modes: List[int],
        max_len: int = 20
    ):
        self.item_seqs = item_seqs
        self.action_seqs = action_seqs
        self.query_seqs = query_seqs
        self.targets = targets
        self.modes = modes
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
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.modes[idx], dtype=torch.long)
        )


class LatentIntentGRU4RecNet(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_dim: int = 64,
        action_dim: int = 16,
        query_dim: int = 50,
        query_proj_dim: int = 32,
        hidden_dim: int = 112,
        bottleneck_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(len(ACTION_MAP) + 1, action_dim, padding_idx=0)
        self.query_proj = nn.Linear(query_dim, query_proj_dim)

        input_dim = item_dim + action_dim + query_proj_dim
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)

        # Informational Latent Intent Bottleneck: compresses [h_t; a_t; q_t] into z_intent
        context_dim = hidden_dim + action_dim + query_proj_dim
        intermediate_dim = 96
        self.intent_bottleneck = nn.Sequential(
            nn.Linear(context_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        # Scoring in shared learned scoring space
        self.fc_catalog = nn.Linear(bottleneck_dim, num_items, bias=True)

        # Auxiliary head for intent mode supervision (0: Discovery, 1: Conversion)
        self.fc_aux_mode = nn.Linear(bottleneck_dim, 2, bias=True)

    def extract_intent(self, items, actions, queries) -> torch.Tensor:
        """Extracts the latent intent vector z_intent."""
        i_emb = self.item_embedding(items)             # [batch, max_len, item_dim]
        a_emb = self.action_embedding(actions)         # [batch, max_len, action_dim]
        q_emb = torch.relu(self.query_proj(queries))   # [batch, max_len, query_proj_dim]

        x = torch.cat([i_emb, a_emb, q_emb], dim=-1)   # [batch, max_len, input_dim]
        x = self.dropout(x)

        _, h_n = self.gru(x)
        h_t = h_n.squeeze(0)                           # [batch, hidden_dim]

        # Extract current step action and query representations
        last_a = a_emb[:, -1, :]                       # [batch, action_dim]
        last_q = q_emb[:, -1, :]                       # [batch, query_proj_dim]

        intent_input = torch.cat([h_t, last_a, last_q], dim=-1) # [batch, context_dim]
        z_intent = self.intent_bottleneck(intent_input)          # [batch, bottleneck_dim]
        return z_intent

    def forward(self, items, actions, queries) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_intent = self.extract_intent(items, actions, queries)
        item_logits = self.fc_catalog(z_intent)
        mode_logits = self.fc_aux_mode(z_intent)
        return item_logits, mode_logits, z_intent


class GRU4RecIntentRecommender:
    def __init__(
        self,
        item_dim: int = 64,
        action_dim: int = 16,
        query_proj_dim: int = 32,
        hidden_dim: int = 112,
        bottleneck_dim: int = 64,
        lambda_aux: float = 0.0,
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
        self.bottleneck_dim = bottleneck_dim
        self.lambda_aux = lambda_aux
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
        label = f"Model 3B (λ={self.lambda_aux})"
        print(f"--> [{label}] Vocabulary: {len(self.item2idx):,} items (min_freq={self.min_freq})")

    def prepare_training_pairs(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        item_seqs = []
        action_seqs = []
        query_seqs = []
        targets = []
        modes = []
        unk_idx = self.item2idx["<UNK>"]

        for sid, events in train_sessions.items():
            seq_items = []
            seq_actions = []
            seq_queries = []
            active_query = np.zeros(50, dtype=np.float32)

            product_indices = []
            for idx, e in enumerate(events):
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
                    prefix_items = seq_items[:tgt_idx]
                    item_seqs.append(prefix_items)
                    action_seqs.append(seq_actions[:tgt_idx])
                    query_seqs.append(seq_queries[:tgt_idx])
                    targets.append(tgt_item)

                    # Discovery (0): target not in seen prefix items; Conversion (1): target in seen prefix items
                    is_conversion = 1 if tgt_item in set(prefix_items) else 0
                    modes.append(is_conversion)

        label = f"Model 3B (λ={self.lambda_aux})"
        print(f"--> [{label}] Generated {len(targets):,} training examples.")
        return item_seqs, action_seqs, query_seqs, targets, modes

    def fit(self, train_sessions: Dict[str, List[Dict[str, Any]]]):
        self.build_vocab(train_sessions)
        item_seqs, action_seqs, query_seqs, targets, modes = self.prepare_training_pairs(train_sessions)

        dataset = IntentSequenceDataset(item_seqs, action_seqs, query_seqs, targets, modes, max_len=20)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = LatentIntentGRU4RecNet(
            num_items=len(self.item2idx),
            item_dim=self.item_dim,
            action_dim=self.action_dim,
            query_dim=50,
            query_proj_dim=self.query_proj_dim,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim
        ).to(self.device)

        criterion_item = nn.CrossEntropyLoss(ignore_index=0)
        criterion_aux = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        label = f"Model 3B (λ={self.lambda_aux})"
        print(f"--> [{label}] Training for {self.epochs} epochs on {self.device}...")
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_item_loss = 0.0
            total_aux_loss = 0.0
            steps = 0

            for b_items, b_actions, b_queries, b_targets, b_modes in loader:
                b_items = b_items.to(self.device)
                b_actions = b_actions.to(self.device)
                b_queries = b_queries.to(self.device)
                b_targets = b_targets.to(self.device)
                b_modes = b_modes.to(self.device)

                optimizer.zero_grad()
                item_logits, mode_logits, _ = self.model(b_items, b_actions, b_queries)
                loss_item = criterion_item(item_logits, b_targets)

                if self.lambda_aux > 0.0:
                    loss_aux = criterion_aux(mode_logits, b_modes)
                    loss = loss_item + self.lambda_aux * loss_aux
                    total_aux_loss += loss_aux.item()
                else:
                    loss = loss_item

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_item_loss += loss_item.item()
                steps += 1

            avg_loss = total_loss / max(1, steps)
            avg_item = total_item_loss / max(1, steps)
            if self.lambda_aux > 0.0:
                avg_aux = total_aux_loss / max(1, steps)
                print(f"    Epoch {epoch}/{self.epochs} - Total Loss: {avg_loss:.4f} (Item: {avg_item:.4f}, Aux: {avg_aux:.4f})")
            else:
                print(f"    Epoch {epoch}/{self.epochs} - Item Loss: {avg_loss:.4f}")

        self.model.eval()

    def _prepare_single_input(self, input_events: List[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], set]:
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
            return None, None, None, seen_items

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

        return t_items, t_actions, t_queries, seen_items

    def extract_representation(self, input_events: List[Dict[str, Any]]) -> np.ndarray:
        """Extracts the 64-D latent intent bottleneck vector z_intent for an input sequence."""
        if self.model is None:
            return np.zeros(self.bottleneck_dim, dtype=np.float32)

        t_items, t_actions, t_queries, _ = self._prepare_single_input(input_events)
        if t_items is None:
            return np.zeros(self.bottleneck_dim, dtype=np.float32)

        with torch.no_grad():
            z_intent = self.model.extract_intent(t_items, t_actions, t_queries)
            return z_intent.squeeze(0).cpu().numpy()

    def predict(self, input_events: List[Dict[str, Any]], top_k: int = 20) -> List[str]:
        if self.model is None:
            return self.popular_items[:top_k]

        t_items, t_actions, t_queries, seen_items = self._prepare_single_input(input_events)
        if t_items is None:
            return [it for it in self.popular_items if it not in seen_items][:top_k]

        with torch.no_grad():
            item_logits, _, _ = self.model(t_items, t_actions, t_queries)
            logits = item_logits.squeeze(0)
            logits[0] = -1e9  # PAD
            logits[1] = -1e9  # UNK

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
