"""
Phase 6 — Unit tests for agent nodes (pure logic, no LLM calls).
Tests: GraphState, ConfidenceScorer logic, DecisionNode routing.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.state import GraphState
from src.agents.nodes.decision import (
    decision_node,
    ROUTE_FINALIZE,
    ROUTE_REWRITE,
    ROUTE_RETRIEVE,
)


# ── GraphState ─────────────────────────────────────────────────────────────────

class TestGraphState:
    def test_default_values(self) -> None:
        state = GraphState(query="What is force majeure?")
        assert state.retry_count == 0
        assert state.max_retries == 2
        assert state.status == "unsafe"
        assert state.avg_confidence == 0.0
        assert state.sub_queries == []
        assert state.claims == []
        assert state.trace == []

    def test_model_copy_with_update(self) -> None:
        state = GraphState(query="test")
        updated = state.model_copy(update={"retry_count": 1, "status": "partial"})
        assert updated.retry_count == 1
        assert updated.status == "partial"
        # Original unchanged
        assert state.retry_count == 0

    def test_trace_appending(self) -> None:
        state = GraphState(query="test")
        step = {"step": "planner", "details": "decomposed", "latency_ms": 50}
        updated = state.model_copy(update={"trace": state.trace + [step]})
        assert len(updated.trace) == 1
        assert updated.trace[0]["step"] == "planner"

    def test_query_required(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GraphState()  # type: ignore


# ── DecisionNode routing ───────────────────────────────────────────────────────

class TestDecisionNode:
    def _make_state(
        self,
        avg_confidence: float,
        retry_count: int = 0,
        max_retries: int = 2,
        status: str = "partial",
    ) -> GraphState:
        return GraphState(
            query="Is this valid?",
            avg_confidence=avg_confidence,
            retry_count=retry_count,
            max_retries=max_retries,
            status=status,
        )

    def test_high_confidence_finalizes(self) -> None:
        """avg_confidence >= 0.8 → always finalize."""
        state = self._make_state(avg_confidence=0.85)
        assert decision_node(state) == ROUTE_FINALIZE

    def test_exact_threshold_finalizes(self) -> None:
        state = self._make_state(avg_confidence=0.8)
        assert decision_node(state) == ROUTE_FINALIZE

    def test_partial_confidence_first_retry_rewrites(self) -> None:
        """0.5 <= avg_confidence < 0.8, retries=0 → rewrite."""
        state = self._make_state(avg_confidence=0.65, retry_count=0)
        assert decision_node(state) == ROUTE_REWRITE

    def test_partial_confidence_retries_exhausted_finalizes(self) -> None:
        """0.5 <= avg_confidence < 0.8, retries=2 → finalize."""
        state = self._make_state(avg_confidence=0.65, retry_count=2)
        assert decision_node(state) == ROUTE_FINALIZE

    def test_low_confidence_first_retry_retrieves(self) -> None:
        """avg_confidence < 0.5, retries=0 → full re-retrieval."""
        state = self._make_state(avg_confidence=0.2, retry_count=0)
        assert decision_node(state) == ROUTE_RETRIEVE

    def test_low_confidence_retries_exhausted_finalizes_unsafe(self) -> None:
        """avg_confidence < 0.5, retries=2 → finalize (unsafe)."""
        state = self._make_state(avg_confidence=0.2, retry_count=2)
        assert decision_node(state) == ROUTE_FINALIZE

    def test_zero_confidence_no_retries_retrieves(self) -> None:
        state = self._make_state(avg_confidence=0.0, retry_count=0)
        assert decision_node(state) == ROUTE_RETRIEVE

    def test_exactly_partial_threshold_rewrites_if_retries_remain(self) -> None:
        state = self._make_state(avg_confidence=0.5, retry_count=1)
        assert decision_node(state) == ROUTE_REWRITE

    def test_max_retries_one_still_works(self) -> None:
        """With max_retries=1, second attempt (retry_count=1) should finalize."""
        state = self._make_state(avg_confidence=0.6, retry_count=1, max_retries=1)
        assert decision_node(state) == ROUTE_FINALIZE


# ── ConfidenceScorer node ──────────────────────────────────────────────────────

class TestConfidenceScorerNode:
    """Tests the confidence scorer logic via direct state manipulation (no LLM)."""

    @pytest.mark.asyncio
    async def test_no_claims_gives_unsafe(self) -> None:
        """Import and test with mocked observability decorator."""
        with patch("src.agents.nodes.confidence_scorer.observe", lambda name: lambda f: f), \
             patch("src.agents.nodes.confidence_scorer.update_span_metadata"):
            from src.agents.nodes.confidence_scorer import confidence_scorer_node
            state = GraphState(query="test", claims=[])
            result = await confidence_scorer_node(state)
            assert result.status == "unsafe"
            assert result.avg_confidence == 0.0

    @pytest.mark.asyncio
    async def test_all_supported_high_confidence_verified(self) -> None:
        with patch("src.agents.nodes.confidence_scorer.observe", lambda name: lambda f: f), \
             patch("src.agents.nodes.confidence_scorer.update_span_metadata"):
            from src.agents.nodes.confidence_scorer import confidence_scorer_node
            claims = [
                {"text": "Claim A", "verdict": "supported", "confidence": 0.9, "sources": []},
                {"text": "Claim B", "verdict": "supported", "confidence": 0.85, "sources": []},
            ]
            state = GraphState(query="test", claims=claims)
            result = await confidence_scorer_node(state)
            assert result.status == "verified"
            assert abs(result.avg_confidence - 0.875) < 1e-6

    @pytest.mark.asyncio
    async def test_mixed_verdicts_gives_partial(self) -> None:
        with patch("src.agents.nodes.confidence_scorer.observe", lambda name: lambda f: f), \
             patch("src.agents.nodes.confidence_scorer.update_span_metadata"):
            from src.agents.nodes.confidence_scorer import confidence_scorer_node
            claims = [
                {"text": "Claim A", "verdict": "supported", "confidence": 0.8, "sources": []},
                {"text": "Claim B", "verdict": "unsupported", "confidence": 0.3, "sources": []},
            ]
            state = GraphState(query="test", claims=claims)
            result = await confidence_scorer_node(state)
            assert result.status in ("partial", "unsafe")

    @pytest.mark.asyncio
    async def test_trace_step_added(self) -> None:
        with patch("src.agents.nodes.confidence_scorer.observe", lambda name: lambda f: f), \
             patch("src.agents.nodes.confidence_scorer.update_span_metadata"):
            from src.agents.nodes.confidence_scorer import confidence_scorer_node
            state = GraphState(query="test", claims=[
                {"text": "C", "verdict": "supported", "confidence": 0.9, "sources": []}
            ])
            result = await confidence_scorer_node(state)
            scorer_steps = [t for t in result.trace if t["step"] == "scorer"]
            assert len(scorer_steps) == 1
            assert "latency_ms" in scorer_steps[0]
