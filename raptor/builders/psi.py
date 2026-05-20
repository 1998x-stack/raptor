from __future__ import annotations

import asyncio
import logging

import numpy as np

from raptor.config import RAPTORConfig
from raptor.tree import TreeNode, UnionFind
from raptor.summarization import summarize_cluster
from raptor.utils import normalize_embeddings

logger = logging.getLogger("raptor.psi")


async def build_tree_psi(
    chunks: list[tuple[str, np.ndarray | list[float]]],
    config: RAPTORConfig,
    llm_chat_fn,
    embed_fn,
    callback=None,
    cancel_check_fn=None,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[int, int]]]:
    """Build a RAPTOR tree using the Psi similarity-merge strategy.

    Instead of iterative clustering, Psi:
        1. Ranks all leaf pairs by cosine similarity.
        2. Merges in that order via union-find to form a tree.
        3. Summarises internal nodes layer by layer (bottom-up).

    Parameters (same as classic builder)."""
    if len(chunks) <= 1:
        return list(chunks), [(0, len(chunks))]

    chunks = list((t, np.asarray(e, dtype=np.float64)) for t, e in chunks)
    layers = [(0, len(chunks))]

    leaves = [
        TreeNode(index=i, text=text, embedding=np.asarray(emb), layer=0)
        for i, (text, emb) in enumerate(chunks)
    ]

    if len(leaves) == 1:
        return chunks, layers

    root, _ = _build_psi_structure(leaves, config)
    root = _rebalance(root, config.max_cluster)

    psi_layers = _collect_layers(root)

    for layer_height in sorted(psi_layers):
        if cancel_check_fn and cancel_check_fn():
            logger.info("RAPTOR Psi build cancelled at layer %d", layer_height)
            break

        nodes = psi_layers[layer_height]
        layer_start = len(chunks)

        async def _summarize_node(node: TreeNode):
            texts = [child.text for child in node.children if child.text]
            if not texts:
                return None
            result = await summarize_cluster(
                texts=texts,
                llm_chat_fn=llm_chat_fn,
                llm_max_length=config.max_token * 10,
                max_token=config.max_token,
                prompt_template=config.summarization_prompt,
                embed_fn=embed_fn,
            )
            if result is None:
                return None
            node.text, node.embedding = result[0], np.asarray(result[1], dtype=np.float64)
            return node

        tasks = [asyncio.create_task(_summarize_node(node)) for node in nodes]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        successful = [n for n in results if n is not None]
        for node in successful:
            chunks.append((node.text, node.embedding))

        if len(chunks) > layer_start:
            layers.append((layer_start, len(chunks)))
            if callback:
                callback(
                    f"Psi layer {layer_height}: {len(nodes)} → "
                    f"{len(chunks) - layer_start} summaries"
                )
        else:
            logger.warning(
                "RAPTOR Psi layer %d produced no summaries; stopping", layer_height
            )
            break

    return chunks, layers


def _build_psi_structure(
    leaves: list[TreeNode], config: RAPTORConfig
) -> tuple[TreeNode, int]:
    n = len(leaves)
    if n <= config.psi_exact_max_leaves:
        return _build_exact(leaves, n)

    buckets = _split_buckets(leaves, config.psi_bucket_size)
    logger.info("Psi bucketed: %d leaves → %d buckets", n, len(buckets))

    bucket_roots = []
    next_id = n
    for bucket in buckets:
        root, next_id = _build_exact(bucket, next_id)
        _assign_prototype(root)
        bucket_roots.append(root)

    if len(bucket_roots) == 1:
        return bucket_roots[0], next_id

    root, next_id = _build_exact(bucket_roots, next_id)
    return root, next_id


def _build_exact(
    nodes: list[TreeNode], next_id: int
) -> tuple[TreeNode, int]:
    n = len(nodes)
    if n == 1:
        return nodes[0], next_id

    ranked = _rank_pairs(nodes)
    uf = UnionFind(n)
    merges = 0
    for i, j in ranked:
        if uf.union(int(i), int(j)):
            merges += 1
        if merges == n - 1:
            break

    tree = uf.tree
    local = {idx: node for idx, node in enumerate(nodes)}
    children_by_parent: dict[int, list[int]] = {}

    for child_idx, parent_idx in enumerate(tree):
        if child_idx not in local:
            local[child_idx] = TreeNode(index=next_id)
            next_id += 1
        if parent_idx == -1:
            continue
        children_by_parent.setdefault(parent_idx, []).append(child_idx)
        if parent_idx not in local:
            local[parent_idx] = TreeNode(index=next_id)
            next_id += 1

    for parent_idx, child_indices in children_by_parent.items():
        parent = local[parent_idx]
        parent.children = [local[c] for c in child_indices]
        for child in parent.children:
            child.parent = parent

    roots = [
        local[idx]
        for idx, p in enumerate(tree)
        if p == -1 and idx in local
    ]
    root = max(roots, key=lambda n: n.index)
    return root, next_id


def _rank_pairs(nodes: list[TreeNode]) -> np.ndarray:
    embeddings = normalize_embeddings(
        np.asarray([n.embedding for n in nodes], dtype=np.float64)
    )
    sim = embeddings @ embeddings.T
    lower = np.tril_indices(len(nodes), -1)
    order = np.argsort(sim[lower], axis=0)[::-1]
    return np.stack([lower[0][order], lower[1][order]], axis=-1)


def _assign_prototype(node: TreeNode) -> np.ndarray:
    if not node.children:
        return np.asarray(node.embedding, dtype=np.float64)
    child_embs = np.asarray([_assign_prototype(c) for c in node.children], dtype=np.float64)
    node.embedding = child_embs.mean(axis=0)
    return node.embedding


def _split_buckets(
    nodes: list[TreeNode], bucket_size: int
) -> list[list[TreeNode]]:
    if len(nodes) <= bucket_size:
        return [nodes]

    embeddings = normalize_embeddings(
        np.asarray([n.embedding for n in nodes], dtype=np.float64)
    )
    groups = [np.arange(len(nodes), dtype=int)]
    buckets = []

    while groups:
        group = groups.pop()
        if len(group) <= bucket_size:
            buckets.append(group.tolist())
            continue

        fanout = min(max(2, int(np.ceil(len(group) / bucket_size))), len(group), 32)
        centers = embeddings[group[:fanout]].copy()

        for _ in range(5):
            scores = embeddings[group] @ centers.T
            labels = np.argmax(scores, axis=1)
            for cid in range(fanout):
                mask = labels == cid
                if np.any(mask):
                    center = embeddings[group][mask].mean(axis=0)
                    norm = np.linalg.norm(center)
                    if norm > 0:
                        centers[cid] = center / norm

        scores = embeddings[group] @ centers.T
        labels = np.argmax(scores, axis=1)
        split_groups = [group[labels == cid].tolist() for cid in range(fanout)]
        split_groups = [g for g in split_groups if g]
        if len(split_groups) <= 1:
            split_groups = [
                group[i : i + bucket_size].tolist()
                for i in range(0, len(group), bucket_size)
            ]
        groups.extend(split_groups)

    result = [[nodes[idx] for idx in bucket] for bucket in buckets if bucket]
    result.sort(key=lambda b: (len(b), b[0].index))
    return result


def _rebalance(root: TreeNode, max_children: int) -> TreeNode:
    max_children = max(2, int(max_children or 2))
    next_id = root.index + 1

    def _walk(node: TreeNode):
        nonlocal next_id
        for child in list(node.children):
            _walk(child)

        while len(node.children) > max_children:
            grouped = []
            for start in range(0, len(node.children), max_children):
                batch = node.children[start : start + max_children]
                if len(batch) == 1:
                    grouped.append(batch[0])
                    batch[0].parent = node
                else:
                    parent = TreeNode(index=next_id, children=batch)
                    next_id += 1
                    for c in batch:
                        c.parent = parent
                    parent.parent = node
                    grouped.append(parent)
            node.children = grouped

    _walk(root)
    while root.parent is not None:
        root = root.parent
    return root


def _collect_layers(root: TreeNode) -> dict[int, list[TreeNode]]:
    layers: dict[int, list[TreeNode]] = {}

    def _height(node: TreeNode) -> int:
        if not node.children:
            return 0
        h = max(_height(c) for c in node.children) + 1
        layers.setdefault(h, []).append(node)
        return h

    _height(root)
    return layers
