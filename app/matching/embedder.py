"""
Sentence-embedding utilities used to find semantically related facts
across different documents.
"""

import logging
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model_cache: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """
    Load (and cache) the embedding model. Loading is slow, so this must
    only happen once per process, not once per fact or comparison.
    """
    global _model_cache
    if _model_cache is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


def fact_text(fact: Dict) -> str:
    """
    Build a single text string representing a fact, used as input to the
    embedding model. Including context fields (unit, time period, scope,
    location) helps the embedding capture meaning, not just the number.
    """
    parts = [
        str(fact.get("subject", "")),
        str(fact.get("predicate", "")),
        str(fact.get("value", "")),
        str(fact.get("unit", "")),
        str(fact.get("time_period", "")),
        str(fact.get("scope", "")),
        str(fact.get("location", "")),
    ]
    return " | ".join(p for p in parts if p)


def embed_facts(facts: List[Dict]) -> np.ndarray:
    """Compute embeddings for a list of facts. Returns an (N, D) numpy array."""
    if not facts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    texts = [fact_text(f) for f in facts]
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)