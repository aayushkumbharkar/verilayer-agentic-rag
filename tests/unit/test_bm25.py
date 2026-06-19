"""
Phase 3 — Unit tests for BM25 search query builder and ranker.
Tests the pure logic (query construction, ranking) without hitting OpenSearch.
"""
from __future__ import annotations

import pytest

from src.retrieval.bm25 import _build_bm25_query
from src.retrieval.ranker import deduplicate_results, rank_results
from src.models.schemas import SearchResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_result(chunk_id: str, score: float, doc_id: str = "doc-1") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=score,
        source="test.pdf",
    )


# ── BM25 Query Builder ─────────────────────────────────────────────────────────

class TestBuildBm25Query:
    def test_basic_structure(self) -> None:
        body = _build_bm25_query("what is negligence", top_k=5)
        assert "query" in body
        assert "size" in body
        assert body["size"] == 5

    def test_multi_match_present(self) -> None:
        body = _build_bm25_query("liability clause", top_k=3)
        bool_query = body["query"]["bool"]
        assert "must" in bool_query
        must = bool_query["must"]
        assert any("multi_match" in clause for clause in must)

    def test_field_boosting(self) -> None:
        body = _build_bm25_query("test", top_k=5)
        multi_match = body["query"]["bool"]["must"][0]["multi_match"]
        # text field should have highest boost
        assert any("text^3" in f for f in multi_match["fields"])

    def test_no_filter_without_filters(self) -> None:
        body = _build_bm25_query("query", top_k=5, filters=None)
        assert "filter" not in body["query"]["bool"]

    def test_filter_applied(self) -> None:
        body = _build_bm25_query("query", top_k=5, filters={"source": "contract.pdf"})
        filter_clauses = body["query"]["bool"]["filter"]
        assert len(filter_clauses) == 1
        assert filter_clauses[0] == {"term": {"source": "contract.pdf"}}

    def test_multiple_filters(self) -> None:
        body = _build_bm25_query("query", top_k=5, filters={"source": "doc.pdf", "section": "3.1"})
        filter_clauses = body["query"]["bool"]["filter"]
        assert len(filter_clauses) == 2

    def test_top_k_reflected_in_size(self) -> None:
        for k in [1, 5, 10, 20]:
            body = _build_bm25_query("q", top_k=k)
            assert body["size"] == k

    def test_fuzziness_auto(self) -> None:
        body = _build_bm25_query("negligence", top_k=5)
        multi_match = body["query"]["bool"]["must"][0]["multi_match"]
        assert multi_match["fuzziness"] == "AUTO"


# ── Ranker ─────────────────────────────────────────────────────────────────────

class TestDeduplicateResults:
    def test_no_duplicates(self) -> None:
        results = [make_result("c1", 0.9), make_result("c2", 0.8)]
        deduped = deduplicate_results(results)
        assert len(deduped) == 2

    def test_keeps_highest_score_duplicate(self) -> None:
        results = [
            make_result("c1", 0.5),
            make_result("c1", 0.9),  # higher score — should win
            make_result("c1", 0.3),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9

    def test_empty_list(self) -> None:
        assert deduplicate_results([]) == []


class TestRankResults:
    def test_sorted_descending(self) -> None:
        results = [
            make_result("c1", 0.3),
            make_result("c2", 0.9),
            make_result("c3", 0.6),
        ]
        ranked = rank_results(results, top_k=3)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_enforced(self) -> None:
        results = [make_result(f"c{i}", float(i) / 10) for i in range(10)]
        ranked = rank_results(results, top_k=3)
        assert len(ranked) == 3

    def test_dedup_then_rank(self) -> None:
        results = [
            make_result("c1", 0.2),
            make_result("c1", 0.8),  # duplicate — higher wins
            make_result("c2", 0.5),
        ]
        ranked = rank_results(results, top_k=5)
        assert len(ranked) == 2
        assert ranked[0].chunk_id == "c1"
        assert ranked[0].score == 0.8

    def test_empty_list(self) -> None:
        assert rank_results([], top_k=5) == []

    def test_fewer_results_than_top_k(self) -> None:
        results = [make_result("c1", 0.9)]
        ranked = rank_results(results, top_k=10)
        assert len(ranked) == 1
