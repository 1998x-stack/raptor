from __future__ import annotations

import logging

import numpy as np

from raptor.config import RAPTORConfig, CLUSTERING_AHC
from raptor.clustering import cluster_gmm, cluster_ahc
from raptor.embedding import reduce_dimensions
from raptor.summarization import summarize_clusters_parallel
from raptor.utils import extract_layer_embeddings, validate_chunks

logger = logging.getLogger("raptor.classic")


async def build_tree_classic(
    chunks: list[tuple[str, np.ndarray | list[float]]],
    config: RAPTORConfig,
    llm_chat_fn,
    embed_fn,
    callback=None,
    cancel_check_fn=None,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[int, int]]]:
    """Build a RAPTOR tree using the classic iterative clustering method.

    Each layer:
        1. UMAP-reduce the current layer's embeddings.
        2. Cluster (GMM or AHC).
        3. Summarise each cluster → new chunks appended.

    Parameters
    ----------
    chunks : list[tuple[str, array]]
        Original (text, embedding) pairs.
    config : RAPTORConfig
    llm_chat_fn : async callable
    embed_fn : async callable
    callback : callable(msg=str), optional
    cancel_check_fn : callable() -> bool, optional
        Return True if the operation should be cancelled.

    Returns
    -------
    enriched_chunks : list[tuple[str, array]]
        Original chunks followed by all generated summaries.
    layers : list[tuple[int, int]]
        ``[(start, end), ...]`` boundaries of each layer in *enriched_chunks*.
    """
    if len(chunks) <= 1:
        return list(chunks), [(0, len(chunks))]

    chunks = validate_chunks(chunks)
    layers = [(0, len(chunks))]
    start, end = 0, len(chunks)

    while end - start > 1:
        if cancel_check_fn and cancel_check_fn():
            logger.info("RAPTOR classic build cancelled at layer %d", len(layers))
            break

        embeddings = extract_layer_embeddings(chunks, start, end)
        n = len(embeddings)

        if n == 2:
            summary_result = await summarize_clusters_parallel(
                clusters=[[0, 1]],
                chunks=chunks,
                offset=start,
                llm_chat_fn=llm_chat_fn,
                llm_max_length=config.max_token * 10,
                max_token=config.max_token,
                prompt_template=config.summarization_prompt,
                embed_fn=embed_fn,
                max_errors=config.max_errors,
                callback=callback,
            )
            produced = len(summary_result)
            if produced == 0:
                logger.warning("RAPTOR layer produced no summaries; stopping")
                break
            chunks.extend(summary_result)
            if callback:
                callback(f"Cluster one layer: {end - start} → {produced}")
            layers.append((end, len(chunks)))
            start, end = end, len(chunks)
            continue

        reduced = reduce_dimensions(
            embeddings,
            random_state=config.random_state,
        )

        if config.clustering_method == CLUSTERING_AHC:
            raw_labels = cluster_ahc(
                reduced,
                max_cluster=config.max_cluster,
            )
            n_clusters = len(np.unique(raw_labels))
            cluster_indices = _labels_to_clusters(raw_labels)
        else:
            soft_labels, n_clusters = cluster_gmm(
                reduced,
                max_cluster=config.max_cluster,
                threshold=config.threshold,
                random_state=config.random_state,
            )
            cluster_indices = _soft_labels_to_clusters(soft_labels, n_clusters)

        if n_clusters <= 1:
            logger.info("RAPTOR: single cluster on layer %d; stopping", len(layers))
            break

        summary_results = await summarize_clusters_parallel(
            clusters=cluster_indices,
            chunks=chunks,
            offset=start,
            llm_chat_fn=llm_chat_fn,
            llm_max_length=config.max_token * 10,
            max_token=config.max_token,
            prompt_template=config.summarization_prompt,
            embed_fn=embed_fn,
            max_errors=config.max_errors,
            callback=callback,
        )

        produced = len(summary_results)
        if produced == 0:
            logger.warning("RAPTOR layer produced no summaries; stopping")
            break
        if produced < n_clusters:
            logger.warning(
                "RAPTOR layer: %d/%d clusters summarised (%d skipped)",
                produced, n_clusters, n_clusters - produced,
            )

        chunks.extend(summary_results)
        if callback:
            callback(f"Cluster one layer: {end - start} → {produced}")
        layers.append((end, len(chunks)))
        start, end = end, len(chunks)

    return chunks, layers


def _labels_to_clusters(labels: np.ndarray) -> list[list[int]]:
    unique = np.unique(labels)
    return [np.where(labels == lbl)[0].tolist() for lbl in unique]


def _soft_labels_to_clusters(
    soft_labels: list[list[int]], n_clusters: int
) -> list[list[int]]:
    clusters = [[] for _ in range(n_clusters)]
    for idx, assigned in enumerate(soft_labels):
        for cluster_id in assigned:
            if 0 <= cluster_id < n_clusters:
                clusters[cluster_id].append(idx)
    return [c for c in clusters if c]
