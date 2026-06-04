from __future__ import annotations
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple


class FAISSItemIndex:
    """
    Builds and queries a FAISS index over item embeddings for fast
    approximate nearest-neighbor retrieval at serving time.
    """

    def __init__(self, embedding_dim: int, index_type: str = "IVFFlat"):
        """
        embedding_dim: dimension of item embeddings
        index_type: one of 'Flat' (exact), 'IVFFlat' (approx), 'HNSW' (graph-based)
        """
        # TODO: self.embedding_dim = embedding_dim
        # TODO: self.index_type = index_type
        # TODO: self.index = None
        # TODO: self.item_ids: List[str] = []  — parallel list mapping FAISS row → item_id
        pass

    def build(
        self,
        item_embeddings: np.ndarray,
        item_ids: List[str],
        nlist: int = 100,
    ) -> None:
        """
        Build the FAISS index from an item embedding matrix.

        item_embeddings: float32 array [num_items, embedding_dim]
        item_ids: list of item id strings, parallel to rows of item_embeddings
        nlist: number of IVF clusters (only used for IVFFlat index type)
        """
        # TODO: assert item_embeddings.shape == (len(item_ids), self.embedding_dim)
        # TODO: L2-normalize embeddings so inner product == cosine similarity:
        #       faiss.normalize_L2(item_embeddings)
        # TODO: build index based on self.index_type:
        #       'Flat'    → faiss.IndexFlatIP(self.embedding_dim)
        #       'IVFFlat' → quantizer = IndexFlatIP; faiss.IndexIVFFlat(quantizer,
        #                   self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT);
        #                   self.index.train(item_embeddings)
        #       'HNSW'    → faiss.IndexHNSWFlat(self.embedding_dim, 32)
        # TODO: self.index.add(item_embeddings)
        # TODO: self.item_ids = item_ids
        # TODO: print(f"FAISS index built: {self.index.ntotal} items, type={self.index_type}")
        pass

    def search(
        self, query_vectors: np.ndarray, k: int = 20
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Retrieve top-k items for each query vector.

        query_vectors: float32 array [num_queries, embedding_dim]
        k: number of results per query
        returns: (scores [num_queries, k], item_id_lists [num_queries][k])
        """
        # TODO: assert self.index is not None, "Call build() before search()"
        # TODO: L2-normalize query_vectors: faiss.normalize_L2(query_vectors)
        # TODO: scores, indices = self.index.search(query_vectors, k)
        # TODO: map integer indices to item_id strings using self.item_ids
        #       (handle index == -1 which FAISS uses for unfilled slots)
        # TODO: return scores, item_id_lists
        pass

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and item_ids list to disk.
        Saves two files: <path>.index and <path>.ids.npy
        """
        # TODO: assert self.index is not None
        # TODO: faiss.write_index(self.index, path + ".index")
        # TODO: np.save(path + ".ids.npy", np.array(self.item_ids))
        # TODO: print(f"FAISS index saved to {path}.index")
        pass

    def load(self, path: str) -> None:
        """
        Load the FAISS index and item_ids from disk.
        Expects <path>.index and <path>.ids.npy to exist.
        """
        # TODO: self.index = faiss.read_index(path + ".index")
        # TODO: self.item_ids = np.load(path + ".ids.npy", allow_pickle=True).tolist()
        # TODO: print(f"FAISS index loaded: {self.index.ntotal} items")
        pass
