"""
conversation_parser.py
======================

This module contains simple functions for extracting entity mentions from
user conversations.  In the POC we use a naive substring check; in a real
system this could be replaced with an LLM‑powered entity extractor.
"""

from __future__ import annotations

from typing import List, Sequence


def extract_entities_from_conversation(conversation: str, known_categories: Sequence[str]) -> List[str]:
    """Returns category identifiers mentioned in the conversation."""
    conv_lower = conversation.lower()
    return [c for c in known_categories if c.lower() in conv_lower]