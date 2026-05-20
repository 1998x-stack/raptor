from __future__ import annotations

import asyncio
import logging

from raptor.utils import truncate_text

logger = logging.getLogger("raptor.summarization")


async def summarize_cluster(
    texts: list[str],
    llm_chat_fn,
    llm_max_length: int,
    max_token: int,
    prompt_template: str,
    embed_fn,
    max_retries: int = 3,
) -> tuple[str, list[float]] | None:
    """Generate a summary for a cluster of text chunks.

    1. Truncate child texts so the combined prompt fits the LLM context.
    2. Call the LLM with retries.
    3. Embed the resulting summary.

    Parameters
    ----------
    texts : list[str]
        Child node texts to summarise.
    llm_chat_fn : async callable
        ``async (system_prompt, messages, gen_config) -> str``.
        Must be an async function that returns the LLM response text.
    llm_max_length : int
        Maximum context length of the LLM (tokens or chars).
    max_token : int
        Maximum response length for the summary.
    prompt_template : str
        Template containing ``{cluster_content}``.
    embed_fn : async callable
        ``async (text: str) -> list[float]``.
    max_retries : int
        Retry count for transient LLM failures.

    Returns
    -------
    (summary_text, embedding) or None
        ``None`` if all retries were exhausted.
    """
    n_chunks = len(texts)
    if n_chunks == 0:
        return None

    chars_per_chunk = max(1, (llm_max_length - max_token) // n_chunks)
    truncated = [truncate_text(t, chars_per_chunk) for t in texts]
    cluster_content = "\n".join(truncated)
    prompt = prompt_template.format(cluster_content=cluster_content)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm_chat_fn(
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                gen_config={"max_tokens": max(max_token, 512)},
            )
            if not response or "**ERROR**" in response:
                raise ValueError(f"LLM returned error marker: {response[:200]}")

            logger.debug("RAPTOR summary (%d chunks) → %d chars", n_chunks, len(response))
            embedding = await embed_fn(response)
            return response, embedding

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "RAPTOR summarization attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(1 + attempt)

    logger.error("RAPTOR summarization failed after %d retries: %s", max_retries, last_exc)
    return None


async def summarize_clusters_parallel(
    clusters: list[list[int]],
    chunks: list[tuple[str, list[float]]],
    offset: int,
    llm_chat_fn,
    llm_max_length: int,
    max_token: int,
    prompt_template: str,
    embed_fn,
    max_retries: int = 3,
    max_errors: int = 3,
    callback = None,
) -> list[tuple[str, list[float]]]:
    """Summarise multiple clusters in parallel and return new (text, emb) pairs.

    Parameters
    ----------
    clusters : list[list[int]]
        Each element is a list of chunk indices (relative to *offset*) that
        form one cluster.
    chunks : list[tuple[str, list[float]]]
        Full chunk list (original chunks + previously added summaries).
    offset : int
        Start index of the current layer in *chunks*.
    max_errors : int
        Abort if this many consecutive summaries fail.
    callback : callable, optional
        ``callback(msg=str)`` for progress reporting.

    Returns
    -------
    new_chunks : list[tuple[str, list[float]]]
        Successfully generated summaries.
    """
    error_count = 0

    async def _summarize_one(cluster_idx: int) -> tuple[str, list[float]] | None:
        nonlocal error_count
        indices = [offset + i for i in clusters[cluster_idx]]
        texts = [chunks[i][0] for i in indices if i < len(chunks)]
        if not texts:
            return None

        result = await summarize_cluster(
            texts=texts,
            llm_chat_fn=llm_chat_fn,
            llm_max_length=llm_max_length,
            max_token=max_token,
            prompt_template=prompt_template,
            embed_fn=embed_fn,
            max_retries=max_retries,
        )
        if result is None:
            error_count += 1
            if callback:
                callback(f"Cluster {cluster_idx} summarization failed")
            if error_count >= max_errors:
                raise RuntimeError(
                    f"RAPTOR aborted after {error_count} consecutive failures"
                )
        return result

    tasks = [_summarize_one(c) for c in range(len(clusters))]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r is not None]
