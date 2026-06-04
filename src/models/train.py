from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import mlflow

from src.models.session_encoder import IntentEncoder, ItemEmbedding
from src.evaluation.metrics import evaluate_all


class SessionDataset(Dataset):
    """PyTorch dataset wrapping processed Coveo sessions for next-item prediction."""

    def __init__(
        self,
        sessions: Dict[str, List[dict]],
        item_vocab: Dict[str, int],
        max_len: int = 20,
    ):
        """
        sessions: processed sessions dict from CoveoDataPipeline
        item_vocab: mapping from item_id string to integer index (1-based; 0 = padding)
        max_len: maximum sequence length — longer sessions are truncated from the left
        """
        # TODO: convert sessions dict to list of (input_sequence, target_item) pairs
        #       input = all events except the last; target = last event's item_id
        # TODO: filter out samples where target item is not in item_vocab (OOV items)
        # TODO: store self.samples as list of (input_item_ids, length, target_idx) tuples
        # TODO: store self.item_vocab and self.max_len
        pass

    def __len__(self) -> int:
        # TODO: return len(self.samples)
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns: (padded_input_tensor [max_len], length, target_item_idx)
        """
        # TODO: retrieve self.samples[idx]
        # TODO: truncate input_item_ids to last max_len items if too long
        # TODO: left-pad with zeros to max_len
        # TODO: return (torch.tensor(padded_ids), length, target_idx)
        pass


def build_item_vocab(sessions: Dict[str, List[dict]]) -> Dict[str, int]:
    """
    Build a mapping from item_id string to integer index from all sessions.
    Returns: dict {item_id: index}, 1-based (0 reserved for padding)
    """
    # TODO: collect all unique item_ids from all session events where item_id is not None
    # TODO: sort for reproducibility
    # TODO: return {item_id: idx+1 for idx, item_id in enumerate(sorted_items)}
    pass


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full training epoch.
    Returns: average cross-entropy loss over all batches
    """
    # TODO: model.train()
    # TODO: iterate batches: (input_ids, lengths, targets) = batch
    # TODO: move input_ids, lengths, targets to device
    # TODO: look up item embeddings for input_ids → [batch, seq_len, embedding_dim]
    # TODO: forward through IntentEncoder → intent_vectors [batch, hidden_dim]
    # TODO: compute dot-product scores against all item embedding weights
    #       scores = intent_vectors @ item_embedding.weight.T → [batch, num_items+1]
    # TODO: compute loss = loss_fn(scores, targets)
    # TODO: optimizer.zero_grad(); loss.backward(); optimizer.step()
    # TODO: accumulate total_loss; return total_loss / len(dataloader)
    pass


def evaluate_epoch(
    encoder: nn.Module,
    item_embedding: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """
    Evaluate next-item prediction on a validation DataLoader.
    Returns: dict from evaluate_all (recall@5/10/20, mrr, ndcg@10)
    """
    # TODO: encoder.eval(); item_embedding.eval()
    # TODO: with torch.no_grad(): iterate batches
    # TODO: compute scores same way as train_epoch
    # TODO: rank all items by score descending; map top-20 indices back to item_id strings
    # TODO: collect all_predictions (list of ranked item id lists) and all_targets
    # TODO: call evaluate_all(all_predictions, all_targets)
    # TODO: return metrics dict
    pass


def train(config: dict) -> None:
    """
    Full training loop with MLflow tracking.

    Required config keys:
        data_dir, processed_dir, model_dir, epochs, lr, batch_size,
        hidden_dim, input_dim, num_layers, dropout, device
    """
    # TODO: mlflow.start_run()
    # TODO: mlflow.log_params(config)
    # TODO: load processed sessions from config['processed_dir'] / 'sessions.pkl'
    # TODO: call build_item_vocab(sessions)
    # TODO: split sessions dict into train (90%) and val (10%) by session_id
    # TODO: create SessionDataset and DataLoader for train and val sets
    # TODO: instantiate IntentEncoder(config['input_dim'], config['hidden_dim'], ...)
    # TODO: instantiate ItemEmbedding(num_items, config['hidden_dim'])
    # TODO: move both models to config['device']
    # TODO: set up Adam optimizer over both models' parameters
    # TODO: set up nn.CrossEntropyLoss
    # TODO: loop epochs:
    #         train_loss = train_epoch(...)
    #         val_metrics = evaluate_epoch(...)
    #         mlflow.log_metrics({train_loss, **val_metrics}, step=epoch)
    #         save checkpoint if val recall@10 improved
    # TODO: mlflow.end_run()
    pass


if __name__ == "__main__":
    config = {
        "data_dir": "data/raw",
        "processed_dir": "data/processed",
        "model_dir": "experiments/checkpoints",
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 128,
        "hidden_dim": 256,
        "input_dim": 384,   # all-MiniLM-L6-v2 output dimension
        "num_layers": 2,
        "dropout": 0.2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    train(config)
