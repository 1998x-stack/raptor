"""
RAPTOR configuration module.

Defines RAPTORConfig: a single dataclass that holds all tunable parameters
for tree building, clustering, summarization, and retrieval.

CleanRL principle: keep all knobs in one place so the user never needs
to hunt across files to understand what can be tuned.
"""

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TREE_BUILDER_CLASSIC: Literal["classic"] = "classic"
TREE_BUILDER_PSI: Literal["psi"] = "psi"
SUPPORTED_TREE_BUILDERS = {TREE_BUILDER_CLASSIC, TREE_BUILDER_PSI}

CLUSTERING_GMM: Literal["gmm"] = "gmm"
CLUSTERING_AHC: Literal["ahc"] = "ahc"
SUPPORTED_CLUSTERING_METHODS = {CLUSTERING_GMM, CLUSTERING_AHC}

RETRIEVAL_TRAVERSE: Literal["tree_traversal"] = "tree_traversal"
RETRIEVAL_COLLAPSED: Literal["collapsed_tree"] = "collapsed_tree"
SUPPORTED_RETRIEVAL_METHODS = {RETRIEVAL_TRAVERSE, RETRIEVAL_COLLAPSED}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RAPTORConfig:
    """All tunable parameters for a RAPTOR pipeline.

    Attributes
    ----------
    max_cluster : int
        Maximum number of clusters per tree layer.  Controls fan-out: fewer
        clusters = broader summaries, more clusters = finer-grained.
    max_token : int
        Maximum tokens for LLM-generated summaries (default 512).
    threshold : float
        GMM soft-clustering probability threshold.  A chunk belongs to every
        cluster whose probability exceeds this value.  Range: (0, 1).
    max_errors : int
        Halt tree-building after this many consecutive LLM failures.
    tree_builder : {"classic", "psi"}
        - "classic":  iterative UMAP → cluster → summarise per layer.
        - "psi":      one-shot similarity merge tree, then layer-by-layer
                      summarisation.  Better for very large corpora.
    clustering_method : {"gmm", "ahc"}
        - "gmm":   Gaussian Mixture with BIC-driven cluster count.
        - "ahc":   Agglomerative Hierarchical Clustering with dendrogram-gap
                   heuristic.
    psi_exact_max_leaves : int
        Psi tree only: build exact for ≤ N leaves; bucket-and-approximate
        beyond.  Default 4096.
    psi_bucket_size : int
        Psi tree only: max leaves per bucket during approximate build.
        Default 1024.  Clamped to psi_exact_max_leaves.
    summarization_prompt : str
        Prompt template for the LLM summariser.  Must contain
        ``{cluster_content}`` which will be replaced with the concatenated
        child-node texts.
    """

    # -- tree / clustering --------------------------------------------------
    max_cluster: int = 10
    max_token: int = 512
    threshold: float = 0.1
    max_errors: int = 3

    # -- builder selection --------------------------------------------------
    tree_builder: Literal["classic", "psi"] = "classic"
    clustering_method: Literal["gmm", "ahc"] = "gmm"

    # -- psi-specific -------------------------------------------------------
    psi_exact_max_leaves: int = 4096
    psi_bucket_size: int = 1024

    # -- summarization ------------------------------------------------------
    summarization_prompt: str = (
        "Write a summary of the following, including as many key "
        "details as possible:\n\n{cluster_content}"
    )

    # -- retrieval defaults -------------------------------------------------
    retrieval_method: Literal["tree_traversal", "collapsed_tree"] = (
        "collapsed_tree"
    )
    retrieval_top_k: int = 5
    retrieval_max_tokens: int = 2000

    # -- random seed --------------------------------------------------------
    random_state: int = 42

    def __post_init__(self):
        """Validate and clamp configuration values."""
        if self.tree_builder not in SUPPORTED_TREE_BUILDERS:
            raise ValueError(
                f"Unsupported tree_builder: {self.tree_builder}. "
                f"Choose from {SUPPORTED_TREE_BUILDERS}."
            )
        if self.clustering_method not in SUPPORTED_CLUSTERING_METHODS:
            raise ValueError(
                f"Unsupported clustering_method: {self.clustering_method}. "
                f"Choose from {SUPPORTED_CLUSTERING_METHODS}."
            )
        if self.retrieval_method not in SUPPORTED_RETRIEVAL_METHODS:
            raise ValueError(
                f"Unsupported retrieval_method: {self.retrieval_method}. "
                f"Choose from {SUPPORTED_RETRIEVAL_METHODS}."
            )

        # clamp Psi params to sensible ranges
        self.psi_exact_max_leaves = max(2, int(self.psi_exact_max_leaves))
        self.psi_bucket_size = min(
            max(2, int(self.psi_bucket_size)),
            self.psi_exact_max_leaves,
        )
        self.max_errors = max(1, int(self.max_errors))
