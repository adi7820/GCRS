"""
demo.py
=======

Entry point for running the modular G‑CRS proof‑of‑concept demonstration.

The script wires together the various modules: it generates synthetic data,
builds a knowledge graph, trains a graph reasoner, simulates a conversation
and outputs recommendations.  It optionally uses a local LLM via the
prompt_builder module.
"""

from __future__ import annotations

import random

from .data_generation import generate_synthetic_data
from .knowledge_graph import build_knowledge_graph, build_extended_knowledge_graph
from .graph_reasoner import GraphReasoner
from .recommender import recommend
from .prompt_builder import build_prompt, generate_llm_response


def run_demo(seed: int = 42, use_extended_graph: bool = True, use_gpu: bool = False) -> None:
    users, items, interactions = generate_synthetic_data(num_users=10, num_items=30, seed=seed)
    # Build either a simple or an extended graph with category/feature nodes
    if use_extended_graph:
        kg = build_extended_knowledge_graph(users, items, interactions)
    else:
        kg = build_knowledge_graph(users, items, interactions)
    reasoner = GraphReasoner(embedding_dim=12)
    reasoner.fit(kg)
    example_user = random.choice(list(users.keys()))
    preferred_cat = users[example_user].preferred_categories[0]
    conversation = f"I recently enjoyed products in {preferred_cat} and I am looking for more recommendations."
    recommended_items, scores_df = recommend(
        user_id=example_user,
        conversation=conversation,
        users=users,
        items=items,
        kg=kg,
        reasoner=reasoner,
        ppr_top_k=15,
        combined_top_k=5,
        alpha=0.85,
        verbose=True,
        use_gpu=use_gpu,
    )
    prompt = build_prompt(example_user, conversation, recommended_items, items, scores_df)
    print("\nGenerated Prompt:\n", prompt)
    # Use local GPT-2 if available
    try:
        response = generate_llm_response(prompt, model_name="gpt2", max_new_tokens=80)
        print("\nLLM Response (GPT-2):\n", response)
    except ImportError:
        print("\nLocal transformers not available; skipping.")
    # Optionally call OpenAI LLM if configured
    try:
        from .prompt_builder import generate_openai_response  # type: ignore
        # You must set OPENAI_API_KEY in environment or pass explicitly
        # response = generate_openai_response(prompt, model="gpt-3.5-turbo")
        # print("\nOpenAI LLM Response:\n", response)
    except Exception:
        pass


if __name__ == "__main__":
    run_demo()