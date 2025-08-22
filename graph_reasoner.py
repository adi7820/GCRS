"""
graph_reasoner.py
=================

This module provides a simple unsupervised graph reasoner based on
truncated SVD of the adjacency matrix.  It mirrors the `GraphReasoner`
class in the monolithic proof‑of‑concept and exposes the same API: fit on
a KnowledgeGraph and obtain candidate items for a user via cosine
similarity in the latent space.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_graph import KnowledgeGraph


class GraphReasoner:
    """Learns low‑dimensional embeddings for graph nodes using SVD."""

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim
        self.node_embeddings: Optional[pd.DataFrame] = None
        self.node_list: List[str] = []

    def fit(self, kg: KnowledgeGraph) -> None:
        A, nodes = kg.to_adjacency_matrix()
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=0)
        latent = svd.fit_transform(A)
        # Normalize embeddings to unit length.  Some rows may have zero norm
        # (e.g. isolated nodes), so we guard against division by zero by
        # replacing zero norms with one.  This ensures no NaNs are produced.
        norms = np.linalg.norm(latent, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        latent = latent / norms
        self.node_embeddings = pd.DataFrame(latent, index=nodes)
        self.node_list = nodes

    def recommend_candidates(self, user_id: str, top_k: int = 10) -> List[str]:
        if self.node_embeddings is None:
            raise ValueError("GraphReasoner.fit must be called before recommendation.")
        if user_id not in self.node_embeddings.index:
            raise ValueError(f"User {user_id} not found in embeddings.")
        user_vec = self.node_embeddings.loc[user_id].values.reshape(1, -1)
        sims = cosine_similarity(user_vec, self.node_embeddings.values)[0]
        candidates = [node for node in self.node_embeddings.index if node.startswith("I")]
        ranked = sorted(((node, sims[self.node_list.index(node)]) for node in candidates), key=lambda x: -x[1])
        return [node for node, _ in ranked[:top_k]]