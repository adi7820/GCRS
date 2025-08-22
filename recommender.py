"""
recommender.py
==============

This module orchestrates the recommendation process by combining the
graph reasoner and personalized PageRank scores.  It imports
components from other modules to perform its task.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .data_generation import User, Item
from .knowledge_graph import KnowledgeGraph
from .graph_reasoner import GraphReasoner
from .conversation_parser import extract_entities_from_conversation


def recommend(
    user_id: str,
    conversation: str,
    users: Dict[str, User],
    items: Dict[str, Item],
    kg: KnowledgeGraph,
    reasoner: GraphReasoner,
    ppr_top_k: int = 10,
    combined_top_k: int = 5,
    alpha: float = 0.85,
    verbose: bool = True,
    use_gpu: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    """Returns a list of recommended items for a user given a conversation."""
    svd_candidates = reasoner.recommend_candidates(user_id, top_k=ppr_top_k)
    # Determine all category nodes from items (names beginning with 'C' but not item IDs).
    known_categories: Sequence[str] = sorted({cat for item in items.values() for cat in item.categories})
    mentioned = extract_entities_from_conversation(conversation, known_categories)
    if verbose:
        print(f"User {user_id} mentions categories: {mentioned}")
    # Build personalization vector.  Allocate half the mass to the user and
    # half to the mentioned category nodes if they exist in the graph.
    personalization = {user_id: 0.5}
    if mentioned:
        cat_nodes = [cat for cat in mentioned if cat in kg.adj]
        if cat_nodes:
            mass = 0.5 / len(cat_nodes)
            for cat in cat_nodes:
                personalization[cat] = mass
    else:
        personalization[user_id] = 1.0
    # Choose between CPU and GPU PageRank
    try:
        if use_gpu:
            ppr_scores = kg.personalized_pagerank_gpu(personalization, alpha=alpha)
        else:
            raise ImportError  # fall back to CPU below
    except ImportError:
        ppr_scores = kg.personalized_pagerank(personalization, alpha=alpha)
    # Filter PPR scores to items only
    item_ppr = {node: score for node, score in ppr_scores.items() if node.startswith("I")}
    if item_ppr:
        max_score = max(item_ppr.values())
        if max_score > 0:
            for node in item_ppr:
                item_ppr[node] /= max_score
    records = []
    for item_id in items:
        svd_score = 1.0 if item_id in svd_candidates else 0.0
        ppr_score = item_ppr.get(item_id, 0.0)
        combined = 0.7 * svd_score + 0.3 * ppr_score
        records.append({
            "item_id": item_id,
            "svd_candidate": svd_score,
            "ppr_score": ppr_score,
            "combined_score": combined,
        })
    scores_df = pd.DataFrame(records).sort_values(by="combined_score", ascending=False)
    recommended_items = scores_df.head(combined_top_k)["item_id"].tolist()
    if verbose:
        print(scores_df.head(combined_top_k))
    return recommended_items, scores_df