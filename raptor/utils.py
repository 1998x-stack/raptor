"""
Shared utility functions for the RAPTOR pipeline.

CleanRL principle: keep helpers small, pure, and well-named.
No hidden state or side effects.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("raptor")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine-similarity operations.

    Zero-vectors remain zero (no NaN propagation).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two sets of normalized embeddings."""
    return np.asarray(a) @ np.asarray(b).T


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------


def extract_layer_embeddings(
    chunks: list[tuple[str, np.ndarray]], start: int, end: int
) -> np.ndarray:
    """Return the embedding matrix for chunks in [start, end)."""
    return np.asarray([chunks[i][1] for i in range(start, end)], dtype=np.float64)


def compute_layer_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Compute the mean embedding (centroid) for each cluster label."""
    unique_labels = np.unique(labels)
    centroids = np.stack(
        [embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels]
    )
    return centroids


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars* characters, appending '…' if shortened."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chunks(chunks: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Filter out chunks with empty text or None embeddings.

    Returns a new list.  Logs a warning if chunks were dropped.
    """
    filtered = [
        (text, emb)
        for text, emb in chunks
        if text and emb is not None and len(emb) > 0
    ]
    dropped = len(chunks) - len(filtered)
    if dropped:
        logger.warning(
            "RAPTOR: dropped %d/%d chunks with empty text or embeddings.",
            dropped,
            len(chunks),
        )
    return filtered
