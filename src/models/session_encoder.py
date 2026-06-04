from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List


class IntentEncoder(nn.Module):
    """
    GRU-based session encoder that produces a dense intent vector
    from a sequence of item/query embeddings within a session.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        input_dim: dimension of input item/query embeddings
        hidden_dim: GRU hidden state dimension (= output intent vector dimension)
        """
        super().__init__()
        # TODO: define self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
        #       dropout=dropout, batch_first=True)
        # TODO: define self.dropout = nn.Dropout(dropout)
        # TODO: define self.projection = nn.Linear(hidden_dim, hidden_dim)
        pass

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: padded input tensor [batch, seq_len, input_dim]
        lengths: actual sequence lengths per batch item [batch]
        returns: intent vectors [batch, hidden_dim]
        """
        # TODO: pack padded sequence: pack_padded_sequence(x, lengths, batch_first=True,
        #       enforce_sorted=False)
        # TODO: pass packed sequence through self.gru
        # TODO: unpack output with pad_packed_sequence
        # TODO: gather the last valid hidden state for each sequence using lengths as index
        # TODO: apply self.dropout
        # TODO: apply self.projection
        # TODO: return intent vectors [batch, hidden_dim]
        pass


class QueryEncoder(nn.Module):
    """
    Wraps sentence-transformers to encode search queries into dense vectors.
    Used during preprocessing to embed raw query strings before session encoding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        # TODO: from sentence_transformers import SentenceTransformer
        # TODO: self.model = SentenceTransformer(model_name)
        # TODO: self.embedding_dim = self.model.get_sentence_embedding_dimension()
        pass

    def encode(self, queries: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        queries: list of raw query strings
        returns: tensor of shape [len(queries), embedding_dim]
        """
        # TODO: call self.model.encode(queries, batch_size=batch_size,
        #       convert_to_tensor=True, show_progress_bar=False)
        # TODO: return tensor on CPU
        pass


class ItemEmbedding(nn.Module):
    """
    Learnable embedding table for item IDs.
    Used alongside query embeddings to represent click events in the session.
    """

    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        # TODO: define self.embedding = nn.Embedding(num_items + 1, embedding_dim,
        #       padding_idx=0)
        # TODO: initialize weights with nn.init.xavier_uniform_(self.embedding.weight)
        pass

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [batch, seq_len] integer tensor (0 = padding)
        returns: [batch, seq_len, embedding_dim]
        """
        # TODO: return self.embedding(item_ids)
        pass
