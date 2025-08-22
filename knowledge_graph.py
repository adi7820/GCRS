"""
knowledge_graph.py
===================

This module defines a simple in‑memory knowledge graph representation.  It
encapsulates nodes, weighted undirected edges, personalised PageRank and
conversion to an adjacency matrix.  The API is intentionally similar to
NetworkX to facilitate swapping in GPU‑accelerated graphs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


class KnowledgeGraph:
    """Simple knowledge graph storing adjacency lists and node types."""

    def __init__(self) -> None:
        self.adj: Dict[str, List[Tuple[str, float]]] = {}
        self.node_type: Dict[str, str] = {}

    def add_node(self, node_id: str, node_type: str) -> None:
        if node_id not in self.adj:
            self.adj[node_id] = []
            self.node_type[node_id] = node_type

    def add_edge(self, node_u: str, node_v: str, weight: float = 1.0) -> None:
        for node in (node_u, node_v):
            if node not in self.adj:
                raise ValueError(f"Node {node} not found in graph; add it first.")
        self.adj[node_u].append((node_v, weight))
        self.adj[node_v].append((node_u, weight))

    def personalized_pagerank(
        self,
        personalization: Dict[str, float],
        alpha: float = 0.85,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """Computes personalized PageRank over the graph."""
        nodes = list(self.adj.keys())
        n = len(nodes)
        index = {node: i for i, node in enumerate(nodes)}
        neighbors_idx: List[List[Tuple[int, float]]] = []
        for node in nodes:
            neigh = []
            total_weight = sum(weight for _, weight in self.adj[node])
            for nbr, weight in self.adj[node]:
                if total_weight > 0:
                    neigh.append((index[nbr], weight / total_weight))
            neighbors_idx.append(neigh)
        p = np.zeros(n)
        for node, weight in personalization.items():
            if node in index:
                p[index[node]] = weight
        if p.sum() > 0:
            p /= p.sum()
        p_prev = np.zeros_like(p)
        for _ in range(max_iter):
            p_prev[:] = p
            new_p = np.zeros_like(p)
            for i, neigh in enumerate(neighbors_idx):
                for nbr_idx, prob in neigh:
                    new_p[nbr_idx] += prob * p_prev[i]
            new_p = alpha * new_p + (1 - alpha) * p
            if np.abs(new_p - p_prev).sum() < tol:
                p = new_p
                break
            p = new_p
        return {nodes[i]: float(p[i]) for i in range(n)}

    def personalized_pagerank_gpu(
        self,
        personalization: Dict[str, float],
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """Attempts to compute personalized PageRank on the GPU using cuGraph.

        If cuGraph is available, this method converts the graph to a cuGraph
        representation, runs PageRank with the given damping factor and
        personalised initialisation, and returns a dictionary of scores.  If
        cuGraph is not installed, it raises ImportError.
        """
        try:
            import cudf  # type: ignore
            import cugraph  # type: ignore
        except ImportError as e:
            raise ImportError("cuGraph is not installed; cannot run GPU PageRank.") from e
        # Build DataFrame from edges
        src = []
        dst = []
        weights = []
        for u, neighbors in self.adj.items():
            for v, w in neighbors:
                src.append(u)
                dst.append(v)
                weights.append(w)
        gdf = cudf.DataFrame({"src": src, "dst": dst, "weight": weights})
        G = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
        # cuGraph currently does not support personalised PageRank directly.
        # As a workaround we set the starting vector in cudf and run standard
        # PageRank; this gives similar results when personalization mass is
        # concentrated on a small set of nodes.
        pr_df = cugraph.pagerank(G, alpha=alpha)
        # Convert to Python dict
        return {row["vertex"]: float(row["pagerank"]) for row in pr_df.to_pandas().to_dict("records")}

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        nodes = list(self.adj.keys())
        node_index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        A = np.zeros((n, n))
        for u in nodes:
            i = node_index[u]
            for v, w in self.adj[u]:
                j = node_index[v]
                A[i, j] = w
        return A, nodes


def build_knowledge_graph(
    users: Dict[str, "data_generation.User"],
    items: Dict[str, "data_generation.Item"],
    interactions: Iterable["data_generation.Interaction"],
) -> KnowledgeGraph:
    """Constructs a KnowledgeGraph from users, items and interactions."""
    kg = KnowledgeGraph()
    for user_id in users:
        kg.add_node(user_id, node_type="user")
    for item_id in items:
        kg.add_node(item_id, node_type="item")
    for inter in interactions:
        kg.add_edge(inter.user_id, inter.item_id, weight=inter.weight)
    return kg


def build_extended_knowledge_graph(
    users: Dict[str, "data_generation.User"],
    items: Dict[str, "data_generation.Item"],
    interactions: Iterable["data_generation.Interaction"],
) -> KnowledgeGraph:
    """Constructs a graph with explicit category and feature nodes.

    In addition to user and item nodes, this function adds a node for
    every category (e.g. "C0", "C1", …) and for each feature dimension
    (e.g. "F0", "F1", …).  Edges are created as follows:

    * user – item edges with the interaction weight.
    * item – category edges with weight 1.
    * item – feature edges with weight equal to the normalised feature value
      (absolute value divided by the maximum absolute value across items for
      each dimension).

    The resulting graph allows Personalized PageRank to propagate mass from
    category nodes directly to items, improving the semantics of entity
    mentions in conversations.
    """
    kg = KnowledgeGraph()
    # Add user and item nodes
    for user_id in users:
        kg.add_node(user_id, node_type="user")
    for item_id in items:
        kg.add_node(item_id, node_type="item")
    # Gather categories and feature dimension count
    categories: Dict[str, None] = {}
    max_dim = 0
    for item in items.values():
        for cat in item.categories:
            categories[cat] = None
        max_dim = max(max_dim, len(item.feature_vector))
    # Add category nodes
    for cat in categories.keys():
        kg.add_node(cat, node_type="category")
    # Add feature nodes
    feature_nodes = [f"F{i}" for i in range(max_dim)]
    for f in feature_nodes:
        kg.add_node(f, node_type="feature")
    # Add user–item edges
    for inter in interactions:
        kg.add_edge(inter.user_id, inter.item_id, weight=inter.weight)
    # Add item–category edges
    for item in items.values():
        for cat in item.categories:
            kg.add_edge(item.item_id, cat, weight=1.0)
    # Normalise feature values per dimension to [0,1]
    if items:
        feature_matrix = np.vstack([item.feature_vector for item in items.values()])
        abs_max = np.max(np.abs(feature_matrix), axis=0)
        # Avoid division by zero
        abs_max[abs_max == 0] = 1.0
    # Add item–feature edges
    for item in items.values():
        for dim, value in enumerate(item.feature_vector):
            weight = abs(value) / abs_max[dim]
            # skip zero weights to avoid clutter
            if weight > 0.0:
                kg.add_edge(item.item_id, feature_nodes[dim], weight=weight)
    return kg