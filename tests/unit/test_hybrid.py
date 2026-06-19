"""
Phase 4 — Unit tests for hybrid search RRF fusion logic.
Tests the pure fusion algorithm without calling OpenSearch or Jina.
"""
from __future__ import annotations

import pytest

from src.retrieval.hybrid import _reciprocal_rank_fusion, RRF_K
from src.models.schemas import SearchResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_result(chunk_id: str, score: float, doc_id: str = "doc-1") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"chunk text for {chunk_id}",
        score=score,
        source="test.pdf",
    )


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_both_empty(self) -> None:
        fused = _reciprocal_rank_fusion([], [])
        assert fused == []

    def test_bm25_only(self) -> None:
        bm25 = [make_result("c1", 0.9), make_result("c2", 0.7)]
        fused = _reciprocal_rank_fusion(bm25, [])
        assert len(fused) == 2
        # c1 ranked first in bm25, so it should have higher RRF score
        chunk_ids = [r.chunk_id for r in fused]
        assert chunk_ids[0] == "c1"

    def test_semantic_only(self) -> None:
        semantic = [make_result("c3", 0.95), make_result("c4", 0.6)]
        fused = _reciprocal_rank_fusion([], semantic)
        assert len(fused) == 2
        assert fused[0].chunk_id == "c3"

    def test_overlap_boosts_score(self) -> None:
        """A chunk appearing in both lists should score higher than one only in BM25."""
        bm25 = [make_result("c1", 0.9), make_result("c2", 0.8)]
        semantic = [make_result("c1", 0.95), make_result("c3", 0.7)]  # c1 overlaps

        fused = _reciprocal_rank_fusion(bm25, semantic)
        chunk_ids = [r.chunk_id for r in fused]
        # c1 appears in both — should be ranked first
        assert chunk_ids[0] == "c1"

    def test_scores_are_positive(self) -> None:
        bm25 = [make_result(f"c{i}", 1.0 / (i + 1)) for i in range(5)]
        semantic = [make_result(f"c{i}", 1.0 / (i + 1)) for i in range(5, 10)]
        fused = _reciprocal_rank_fusion(bm25, semantic)
        for r in fused:
            assert r.score > 0.0

    def test_rrf_formula_correct(self) -> None:
        """Verify RRF formula: weight * 1 / (k + rank) — score is rounded to 6 dp."""
        bm25 = [make_result("only", 1.0)]
        fused = _reciprocal_rank_fusion(bm25, [], bm25_weight=0.5)
        expected_score = round(0.5 * (1.0 / (RRF_K + 1)), 6)
        assert fused[0].score == expected_score

    def test_no_duplicates_in_output(self) -> None:
        """Each chunk_id should appear only once in fused output."""
        bm25 = [make_result("c1", 0.9), make_result("c2", 0.7)]
        semantic = [make_result("c1", 0.8), make_result("c3", 0.6)]
        fused = _reciprocal_rank_fusion(bm25, semantic)
        chunk_ids = [r.chunk_id for r in fused]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_custom_weights(self) -> None:
        """Higher BM25 weight should favour BM25-top result."""
        bm25 = [make_result("bm25_top", 1.0)]
        semantic = [make_result("sem_top", 1.0)]
        fused = _reciprocal_rank_fusion(bm25, semantic, bm25_weight=0.9, semantic_weight=0.1)
        assert fused[0].chunk_id == "bm25_top"

    def test_output_sorted_descending(self) -> None:
        bm25 = [make_result(f"c{i}", 1.0 / (i + 1)) for i in range(5)]
        semantic = [make_result(f"s{i}", 1.0 / (i + 1)) for i in range(5)]
        fused = _reciprocal_rank_fusion(bm25, semantic)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_correct_text(self) -> None:
        """Source metadata from the first list (BM25) should be preserved."""
        bm25 = [make_result("c1", 0.9)]
        semantic = [make_result("c1", 0.8)]  # same chunk, different object
        fused = _reciprocal_rank_fusion(bm25, semantic)
        assert fused[0].chunk_id == "c1"
        assert fused[0].text == "chunk text for c1"
