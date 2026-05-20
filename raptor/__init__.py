"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

A tree-based retrieval system that augments LLM capabilities by recursively
clustering and summarizing text chunks into a hierarchical tree structure.

Reference:
    Sarthi et al., "RAPTOR: Recursive Abstractive Processing for
    Tree-Organized Retrieval", ICLR 2024.

Usage:
    from raptor import RAPTORConfig, RAPTORBuilder, RAPTORRetriever

    config = RAPTORConfig(max_cluster=10, tree_builder="classic")
    builder = RAPTORBuilder(config)
    chunks, layers = await builder.build(embedded_chunks)
    retriever = RAPTORRetriever(chunks, layers)
    results = retriever.retrieve(query_embedding, top_k=5)
"""

from raptor.config import RAPTORConfig
from raptor.tree import TreeNode, UnionFind
from raptor.clustering import cluster_gmm, cluster_ahc
from raptor.embedding import reduce_dimensions
from raptor.summarization import summarize_cluster
from raptor.retrieval import tree_traverse, collapsed_tree
from raptor.builders.classic import ClassicBuilder
from raptor.builders.psi import PsiBuilder

__all__ = [
    "RAPTORConfig",
    "TreeNode",
    "UnionFind",
    "cluster_gmm",
    "cluster_ahc",
    "reduce_dimensions",
    "summarize_cluster",
    "tree_traverse",
    "collapsed_tree",
    "ClassicBuilder",
    "PsiBuilder",
]
