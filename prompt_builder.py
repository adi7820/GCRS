"""
prompt_builder.py
=================

Functions to create prompts from recommendation context and optionally
generate responses using a Hugging Face LLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .data_generation import Item


def build_prompt(
    user_id: str,
    conversation: str,
    recommended_items: List[str],
    items: Dict[str, Item],
    scores_df: pd.DataFrame,
    max_context_items: int = 5,
) -> str:
    """Constructs a prompt summarising candidate items and scores."""
    lines = []
    lines.append("You are a conversational recommender system. Given the user's conversation and a set of candidate items, you must suggest the best items and explain your reasoning.")
    lines.append(f"User ID: {user_id}")
    lines.append(f"Conversation: {conversation}\n")
    lines.append("Candidate Items:")
    for item_id in recommended_items[:max_context_items]:
        item = items[item_id]
        svd_score = scores_df.loc[scores_df["item_id"] == item_id, "svd_candidate"].iloc[0]
        ppr_score = scores_df.loc[scores_df["item_id"] == item_id, "ppr_score"].iloc[0]
        combined = scores_df.loc[scores_df["item_id"] == item_id, "combined_score"].iloc[0]
        lines.append(f"- {item_id} (categories: {', '.join(item.categories)}) | SVD candidate: {svd_score:.2f}, PPR: {ppr_score:.2f}, combined: {combined:.2f}")
    lines.append("\nInstructions: Use the candidate information to recommend up to three items to the user. Explain why each item is appropriate based on the user's preferences and the graph scores. Respond in a friendly and concise manner.")
    return "\n".join(lines)


def generate_llm_response(prompt: str, model_name: str = "gpt2", max_new_tokens: int = 100) -> str:
    """Generates a response from a Hugging Face causal language model.

    The transformers and torch libraries are imported lazily to avoid hard
    dependencies.  If these packages are missing, an ImportError will be
    raised.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers or torch is not installed. Please install them to use LLM generation."
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
    return generated_text


def generate_openai_response(prompt: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None) -> str:
    """Generates a response using an OpenAI chat model.

    Parameters
    ----------
    prompt : str
        The instruction and context to send to the model.
    model : str, default='gpt-3.5-turbo'
        Name of the OpenAI model to use.  See OpenAI documentation for
        available models.
    api_key : str, optional
        OpenAI API key.  If ``None``, the function will look for the
        ``OPENAI_API_KEY`` environment variable.

    Returns
    -------
    str
        The assistant's reply.
    """
    try:
        import os
        import openai  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The openai package is required to use generate_openai_response."
        ) from e
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("An OpenAI API key must be provided via argument or OPENAI_API_KEY env var.")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message["content"].strip()