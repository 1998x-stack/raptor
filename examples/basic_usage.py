#!/usr/bin/env python3
"""
Minimal end-to-end example of RAPTOR tree building and retrieval.

This example uses:
  - Synthetic text chunks with random embeddings (stand-in for real data).
  - A mock LLM that returns a stub summary (replace with your own).
  - A mock embedder that returns the input as a vector (replace with yours).

Requirements:
    pip install raptor numpy scikit-learn umap-learn

To adapt for real use:
    1. Replace ``mock_llm_chat`` with an async call to your LLM (OpenAI, etc.).
    2. Replace ``mock_embed`` with your embedding model.
    3. Provide real text chunks with real embeddings.
"""

import asyncio
import numpy as np

from raptor.config import RAPTORConfig
from raptor.builders.classic import build_tree_classic
from raptor.builders.psi import build_tree_psi
from raptor.retrieval import collapsed_tree


# ---------------------------------------------------------------------------
# Mock LLM & embedder
# ---------------------------------------------------------------------------

async def mock_llm_chat(system: str, messages: list, gen_config: dict) -> str:
    """Stub summariser: concatenates child texts into a short summary."""
    content = messages[0]["content"] if messages else ""
    words = content.split()
    return " ".join(words[: min(30, len(words))]) + " ..."


async def mock_embed(text: str) -> list[float]:
    """Stub embedder: returns a deterministic pseudo-random vector."""
    rng = np.random.RandomState(hash(text) % (2**31))
    return rng.randn(384).tolist()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_chunks(n: int = 12, seed: int = 42) -> list[tuple[str, np.ndarray]]:
    """Generate *n* text chunks with random embeddings for testing."""
    rng = np.random.RandomState(seed)
    topics = ["apples", "oranges", "bananas", "grapes", "peaches"]
    chunks = []
    for i in range(n):
        topic = topics[i % len(topics)]
        text = f"Chunk {i} about {topic}: " + " ".join(
            f"detail_{i}_{j}" for j in range(rng.randint(3, 8))
        )
        emb = rng.randn(384).astype(np.float64)
        chunks.append((text, emb))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    chunks = make_synthetic_chunks(n=12)

    # ----- Classic builder -----
    config_classic = RAPTORConfig(
        max_cluster=5,
        max_token=128,
        tree_builder="classic",
        clustering_method="gmm",
    )

    print("Building RAPTOR tree (classic) ...")
    enriched, layers = await build_tree_classic(
        chunks=chunks,
        config=config_classic,
        llm_chat_fn=mock_llm_chat,
        embed_fn=mock_embed,
        callback=lambda msg: print(f"  [callback] {msg}"),
    )

    print(f"\nClassic tree: {len(enriched)} total chunks, {len(layers)} layers")
    for i, (s, e) in enumerate(layers):
        print(f"  Layer {i}: indices [{s}, {e})  size={e - s}")

    # ----- Psi builder -----
    config_psi = RAPTORConfig(
        max_cluster=4,
        max_token=128,
        tree_builder="psi",
    )

    print("\nBuilding RAPTOR tree (psi) ...")
    enriched_psi, layers_psi = await build_tree_psi(
        chunks=chunks,
        config=config_psi,
        llm_chat_fn=mock_llm_chat,
        embed_fn=mock_embed,
    )

    print(f"Psi tree: {len(enriched_psi)} total chunks, {len(layers_psi)} layers")

    # ----- Retrieval -----
    query_emb = list(np.random.RandomState(99).randn(384))
    results = collapsed_tree(
        query_embedding=query_emb,
        layers=layers,
        chunks=enriched,
        top_k=3,
    )
    print(f"\nCollapsed tree retrieval (top 3):")
    for idx in results:
        text_preview = enriched[idx][0][:80]
        print(f"  chunk[{idx}]: {text_preview}...")


if __name__ == "__main__":
    asyncio.run(main())
