"""
VeriLayer — Phase 3: Result Ranker.
Post-processes search results: deduplicates, re-scores, and trims to top_k.
"""
from __future__ import annotations

import structlog

from src.models.schemas import SearchResult

logger = structlog.get_logger("verilayer.retrieval.ranker")


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate chunks (same chunk_id), keeping highest-scored copy."""
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
            seen[r.chunk_id] = r
    return list(seen.values())


def rank_results(
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Deduplicate, sort by score descending, and trim to top_k.

    Args:
        results: Raw search results (may have duplicates).
        top_k: Maximum results to return.

    Returns:
        Sorted, deduplicated, trimmed list.
    """
    deduped = deduplicate_results(results)
    ranked = sorted(deduped, key=lambda r: r.score, reverse=True)
    trimmed = ranked[:top_k]

    logger.info(
        "results_ranked",
        input_count=len(results),
        deduped_count=len(deduped),
        output_count=len(trimmed),
    )
    return trimmed
