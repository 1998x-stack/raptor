from __future__ import annotations

import numpy as np


def reduce_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int | None = None,
    n_components: int | None = None,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensionality using UMAP.

    UMAP preserves local neighbourhood structure, making it easier for
    clustering algorithms to find semantically coherent groups in the
    reduced space.  This counteracts the "curse of dimensionality" where
    distance metrics become less meaningful in high-dimensional spaces.

    Parameters
    ----------
    embeddings : shape (n_samples, n_features)
        High-dimensional embedding vectors.
    n_neighbors : int, optional
        UMAP n_neighbors parameter.  Defaults to ``(n-1)**0.8``,
        which adapts to the dataset size.
    n_components : int, optional
        Target dimensionality.  Defaults to ``min(12, n-2)``.
    metric : str
        Distance metric for UMAP.  Default ``"cosine"``.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    reduced : shape (n_samples, n_components)
        Low-dimensional representation.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "UMAP is required for dimensionality reduction. "
            "Install with: pip install umap-learn"
        )

    n = len(embeddings)
    if n <= 2:
        return np.asarray(embeddings, dtype=np.float64)

    if n_neighbors is None:
        n_neighbors = max(2, int((n - 1) ** 0.8))
    if n_components is None:
        n_components = min(12, n - 2)

    n_neighbors = max(2, min(n_neighbors, n - 1))
    n_components = max(1, min(n_components, n - 1))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(np.asarray(embeddings, dtype=np.float64))
