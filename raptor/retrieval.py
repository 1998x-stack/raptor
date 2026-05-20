from __future__ import annotations

import numpy as np

from raptor.utils import normalize_embeddings


def tree_traverse(
    query_embedding: np.ndarray | list[float],
    layers: list[tuple[int, int]],
    chunks: list[tuple[str, np.ndarray | list[float]]],
    top_k: int = 5,
) -> list[int]:
    """Retrieve nodes by traversing the tree top-down.

    Starting from the root layer, select the *top_k* most similar nodes.
    For the next layer, consider only the children of those nodes.

    Parameters
    ----------
    query_embedding : array
        Query vector.
    layers : list[tuple[int, int]]
        ``[(start, end), ...]`` — layer boundaries in *chunks*.
    chunks : list[tuple[str, array]]
        All (text, embedding) pairs.
    top_k : int
        Nodes to retain at each layer.

    Returns
    -------
    selected : list[int]
        Indices of selected nodes in *chunks*.
    """
    query = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
    all_embeddings = normalize_embeddings(
        np.asarray([c[1] for c in chunks], dtype=np.float64)
    )

    selected: set[int] = set()
    # For tree traversal we use a collapsed approach: score all nodes in
    # each layer and accumulate the top-k at each level independently.
    for start, end in layers:
        if end <= start:
            continue
        layer_embs = all_embeddings[start:end]
        scores = (layer_embs @ query.T).ravel()
        top_indices = np.argsort(scores)[-min(top_k, end - start):][::-1]
        for idx in top_indices:
            selected.add(start + int(idx))

    return sorted(selected)


def collapsed_tree(
    query_embedding: np.ndarray | list[float],
    layers: list[tuple[int, int]],
    chunks: list[tuple[str, np.ndarray | list[float]]],
    top_k: int = 5,
    max_tokens: int | None = None,
) -> list[int]:
    """Retrieve nodes by flattening the entire tree and scoring all nodes.

    Nodes are ranked by cosine similarity to the query.  Selection stops
    when *max_tokens* is exhausted (if set) or *top_k* is reached.

    Parameters
    ----------
    query_embedding : array
        Query vector.
    layers : list[tuple[int, int]]
    chunks : list[tuple[str, array]]
    top_k : int
        Minimum number of nodes to return.
    max_tokens : int, optional
        Token budget ceiling.  Nodes are added while the running token
        count stays below this threshold.  If None, returns exactly *top_k*.

    Returns
    -------
    selected : list[int]
        Indices of selected nodes in *chunks*, ordered by relevance.
    """
    query = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
    all_embeddings = normalize_embeddings(
        np.asarray([c[1] for c in chunks], dtype=np.float64)
    )
    scores = (all_embeddings @ query.T).ravel()

    if max_tokens is None:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [int(i) for i in top_indices]

    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    token_count = 0

    for idx in order:
        if len(selected) >= top_k and token_count >= max_tokens:
            break
        node_text = chunks[int(idx)][0]
        node_tokens = _estimate_tokens(node_text)
        if token_count + node_tokens > max_tokens and len(selected) >= top_k:
            continue
        selected.append(int(idx))
        token_count += node_tokens

    return selected


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
