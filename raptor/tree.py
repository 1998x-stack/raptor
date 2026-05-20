"""
Tree data structures for RAPTOR.

CleanRL principle: provide simple, well-documented data containers
rather than deeply nested OOP hierarchies.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the RAPTOR tree.

    Leaf nodes hold original text chunks and their embeddings.  Internal
    nodes hold LLM-generated summaries and averaged embeddings.

    Attributes
    ----------
    index : int
        Unique identifier for this node within the tree.
    text : str
        Node text content (original chunk or generated summary).
    embedding : np.ndarray | None
        Dense vector embedding of the node text.
    children : list[TreeNode]
        Child nodes (empty for leaf nodes).
    parent : TreeNode | None
        Parent node reference (None for the root).
    layer : int
        Tree layer: 0 = leaves, 1 = first summary layer, etc.
    is_leaf : bool
        True if this node has no children.
    """

    index: int
    text: str = ""
    embedding: np.ndarray | None = None
    children: list["TreeNode"] = field(default_factory=list)
    parent: "TreeNode | None" = None
    layer: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def height(self) -> int:
        """Height from this node to the deepest leaf (0 for leaves)."""
        if self.is_leaf:
            return 0
        return 1 + max(child.height for child in self.children)

    def iter_nodes(self):
        """Depth-first iterator over all nodes in the subtree."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    def __repr__(self) -> str:
        leaf_info = "leaf" if self.is_leaf else f"{len(self.children)} children"
        text_preview = self.text[:40].replace("\n", " ") if self.text else "<empty>"
        return (
            f"TreeNode(idx={self.index}, layer={self.layer}, "
            f"{leaf_info}, text='{text_preview}...')"
        )


# ---------------------------------------------------------------------------
# UnionFind (for Psi tree construction)
# ---------------------------------------------------------------------------


class UnionFind:
    """Union-Find with ranks for building the Psi merge tree.

    In the Psi algorithm, leaf nodes are merged in order of decreasing
    embedding similarity.  This UnionFind records each merge in a compact
    ``child -> parent`` array that can later be materialized into a
    TreeNode hierarchy.

    Parameters
    ----------
    n : int
        Number of leaf nodes.

    Attributes
    ----------
    tree : list[int]
        ``tree[child_id] = parent_id``.  -1 means the node is a root.
        The array size is at most ``2*n - 1``.
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("UnionFind requires at least 1 leaf node.")

        # Per-leaf rank (for union-by-rank).
        self._rank: list[int] = [0] * n
        # Ancestor chain per leaf (lazily built).
        self._parent_chains: list[list[int]] = [[] for _ in range(n)]
        # All node-ids that ever represented leaf i.
        self._node_ids: list[list[int]] = [[i] for i in range(n)]
        # Compact child→parent array. -1 = root.
        self._tree: list[int] = [-1] * max(1, 2 * n - 1)
        self._next_id: int = n
        self._merge_count: int = 0

    # ---- public API -------------------------------------------------------

    @property
    def tree(self) -> list[int]:
        """Child-to-parent array (size = number of created nodes)."""
        return self._tree[: self._next_id]

    @property
    def merge_count(self) -> int:
        """How many successful merges have been performed."""
        return self._merge_count

    def union(self, i: int, j: int) -> bool:
        """Attempt to merge leaves *i* and *j*.

        Returns True if a new edge was added, False if they are already
        in the same tree.
        """
        root_i = self._representative(i)
        root_j = self._representative(j)
        if root_i == root_j:
            return False

        if self._rank[root_i] < self._rank[root_j]:
            self._merge_lower_into_higher(root_i, root_j, i, j)
        elif self._rank[root_i] > self._rank[root_j]:
            self._merge_lower_into_higher(root_j, root_i, j, i)
        else:
            self._merge_equal_rank(root_i, root_j, i)

        self._merge_count += 1
        return True

    def materialize(self) -> "TreeNode":
        """Convert the internal parent array into a TreeNode tree.

        Returns the root TreeNode.
        """
        from raptor.tree import TreeNode  # local import to avoid circular

        n_leaves = self._next_id
        nodes: dict[int, TreeNode] = {}

        for child_idx in range(n_leaves):
            parent_idx = self._tree[child_idx]
            if child_idx not in nodes:
                nodes[child_idx] = TreeNode(index=child_idx)
            if parent_idx == -1:
                continue
            if parent_idx not in nodes:
                nodes[parent_idx] = TreeNode(index=parent_idx)

            parent = nodes[parent_idx]
            child = nodes[child_idx]
            child.parent = parent
            parent.children.append(child)

        # find the root (node whose parent is -1)
        root_candidates = [
            nodes[idx]
            for idx in range(n_leaves)
            if self._tree[idx] == -1 and idx in nodes
        ]
        if not root_candidates:
            raise RuntimeError("UnionFind tree has no root node.")
        # among candidates, pick the one with the highest index (last created)
        root = max(root_candidates, key=lambda n: n.index)
        return root

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _ordered_extend(target: list, values: list):
        """Extend *target* with unseen values, preserving order."""
        for v in values:
            if v not in target:
                target.append(v)

    def _find(self, i: int) -> list[int]:
        """Lazily compute the ancestor chain for leaf *i*."""
        chain = self._parent_chains[i]
        if not chain or (len(chain) == 1 and chain[0] == i):
            return [i]
        if chain[0] == i:
            self._ordered_extend(chain, self._find(chain[1]))
        else:
            self._ordered_extend(chain, self._find(chain[0]))
        return chain

    def _representative(self, i: int) -> int:
        """Return the root node id for leaf *i*."""
        return self._find(i)[-1]

    def _rank_bisect_right(self, chain: list[int], rank: int) -> int:
        """First chain index whose rank exceeds *rank*."""
        idx = 0
        while idx < len(chain) and self._rank[chain[idx]] <= rank:
            idx += 1
        return idx

    def _add_edge(self, child: int, parent: int):
        """Record ``child -> parent``."""
        self._tree[child] = parent

    def _new_internal_node(self) -> int:
        """Allocate a fresh internal node id."""
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def _merge_lower_into_higher(
        self, lower_root: int, higher_root: int, lower_leaf: int, higher_leaf: int
    ):
        """Merge lower-ranked tree into a specific point of the higher tree."""
        if not self._parent_chains[higher_root]:
            self._parent_chains[higher_root].append(higher_root)
        chain = self._parent_chains[higher_leaf]
        insert_idx = self._rank_bisect_right(chain, self._rank[lower_root])
        insert_idx = min(insert_idx, len(chain) - 1)
        insert_point = chain[insert_idx]
        self._ordered_extend(self._parent_chains[lower_root], chain[insert_idx:])
        # wire the lower root's top node to the insertion point
        lower_top = self._node_ids[lower_leaf][-1]
        parent_ids = self._node_ids[insert_point]
        rank_idx = min(self._rank[lower_leaf] + 1, len(parent_ids) - 1)
        self._add_edge(lower_top, parent_ids[rank_idx])

    def _merge_equal_rank(self, root_i: int, root_j: int, leaf_i: int):
        """Merge two trees of equal rank under a new node."""
        if not self._parent_chains[root_i]:
            self._parent_chains[root_i].append(root_i)
        self._ordered_extend(self._parent_chains[root_j], self._parent_chains[root_i][-1:])
        self._rank[root_i] += 1

        new_id = self._new_internal_node()
        top_i = self._node_ids[leaf_i][-1]
        # find leaf j's representative leaf
        leaf_j = next(
            k for k, ids in enumerate(self._node_ids)
            if ids and ids[0] == root_j
        )
        top_j = self._node_ids[leaf_j][-1]

        self._add_edge(top_i, new_id)
        self._add_edge(top_j, new_id)
        self._node_ids[leaf_i].append(new_id)
