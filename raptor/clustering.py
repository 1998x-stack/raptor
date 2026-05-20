from __future__ import annotations

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

logger = logging.getLogger("raptor.clustering")


def cluster_gmm(
    embeddings: np.ndarray,
    max_cluster: int,
    threshold: float = 0.1,
    random_state: int = 42,
) -> tuple[list[list[int]], int]:
    """Cluster embeddings with a Gaussian Mixture Model.

    The number of components is chosen automatically by minimising the
    Bayesian Information Criterion (BIC).  Soft assignment is used: a
    sample belongs to every cluster whose probability exceeds *threshold*.

    Parameters
    ----------
    embeddings : shape (n, d)
        Input vectors (typically UMAP-reduced).
    max_cluster : int
        Upper bound on the number of components to evaluate.
    threshold : float
        Probability threshold for soft assignment.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    labels : list[list[int]]
        ``labels[sample_idx]`` = list of cluster indices the sample belongs to.
    n_clusters : int
        Number of clusters used.
    """
    n = len(embeddings)
    if n <= 1:
        return [[0] for _ in range(n)], 1

    max_k = min(max_cluster, n)
    if max_k <= 1:
        return [[0] for _ in range(n)], 1

    n_components_range = np.arange(1, max_k + 1)
    bics = []
    for k in n_components_range:
        gm = GaussianMixture(n_components=k, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    best_k = int(n_components_range[np.argmin(bics)])
    logger.info("GMM BIC selected n_clusters=%d (max=%d)", best_k, max_k)

    if best_k <= 1:
        return [[0] for _ in range(n)], 1

    gm = GaussianMixture(n_components=best_k, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)

    labels = []
    for prob in probs:
        assigned = [int(c) for c in np.where(prob > threshold)[0]]
        if not assigned:
            assigned = [int(np.argmax(prob))]
        labels.append(assigned)

    return labels, best_k


def cluster_ahc(
    embeddings: np.ndarray,
    max_cluster: int,
    max_refine_iter: int = 5,
) -> np.ndarray:
    """Cluster embeddings with Agglomerative Hierarchical Clustering.

    Uses Ward linkage and a dendrogram-gap heuristic to choose the
    cluster count, followed by iterative centroid-based refinement.

    Parameters
    ----------
    embeddings : shape (n, d)
        Input vectors (typically UMAP-reduced).
    max_cluster : int
        Upper bound on the number of clusters.
    max_refine_iter : int
        Maximum refinement iterations.

    Returns
    -------
    labels : shape (n,) array of int
        Hard cluster assignment for each sample.
    """
    n = len(embeddings)
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n == 2:
        return np.arange(n, dtype=int)

    full = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        compute_distances=True,
        linkage="ward",
    )
    full.fit(embeddings)

    distances = full.distances_
    if len(distances) > 1:
        gaps = np.diff(distances)
        max_gap = int(np.argmax(gaps))
        n_clusters = max(1, min(n - max_gap - 1, max_cluster))
    else:
        n_clusters = max(1, min(n, max_cluster))

    logger.info("AHC dendrogram-gap selected n_clusters=%d (max=%d)", n_clusters, max_cluster)

    if n_clusters <= 1:
        return np.zeros(n, dtype=int)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    raw_labels = clustering.fit_predict(embeddings)
    refined = _refine_labels(embeddings, raw_labels, max_refine_iter)
    return refined


def _refine_labels(
    embeddings: np.ndarray, labels: np.ndarray, max_iter: int
) -> np.ndarray:
    """Iteratively reassign points to nearest cluster centroid.

    This post-processes AHC output to reduce boundary errors.
    """
    labels = labels.copy()
    for _ in range(max_iter):
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return labels

        centroids = np.stack(
            [embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels]
        )
        diffs = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dists = (diffs ** 2).sum(axis=2)
        new_indices = np.argmin(sq_dists, axis=1)
        new_labels = unique_labels[new_indices]

        if np.array_equal(new_labels, labels):
            break

        remap = {old: new for new, old in enumerate(np.unique(new_labels))}
        labels = np.array([remap[int(lbl)] for lbl in new_labels])

    return labels
