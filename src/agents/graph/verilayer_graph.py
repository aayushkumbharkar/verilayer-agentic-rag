"""
VeriLayer — Phase 6: LangGraph graph assembly.
Wires all 9 agent nodes into the agentic verification pipeline.

Pipeline flow:
  planner → retriever → grader → generator → extractor → verifier → scorer
      └── decision ──┬──→ finalize (END)
                     ├──→ rewriter → verifier (retry rewrite)
                     └──→ retriever (retry with expanded query)
"""
from __future__ import annotations

import time
import structlog
from langgraph.graph import StateGraph, END

from src.agents.state import GraphState
from src.agents.nodes.planner import planner_node
from src.agents.nodes.sub_query_retriever import sub_query_retriever_node
from src.agents.nodes.document_grader import document_grader_node
from src.agents.nodes.generator import generator_node
from src.agents.nodes.claim_extractor import claim_extractor_node
from src.agents.nodes.claim_verifier import claim_verifier_node
from src.agents.nodes.confidence_scorer import confidence_scorer_node
from src.agents.nodes.rewriter import rewriter_node
from src.agents.nodes.decision import (
    decision_node,
    ROUTE_FINALIZE,
    ROUTE_REWRITE,
    ROUTE_RETRIEVE,
)

logger = structlog.get_logger("verilayer.graph")


# ── Finalize node (terminal) ──────────────────────────────────────────────────

async def finalize_node(state: GraphState) -> GraphState:
    """
    Terminal node: compute total latency and seal final status.
    The status was already set by confidence_scorer; just record timing.
    """
    total_latency = sum(s.get("latency_ms", 0) for s in state.trace)
    logger.info(
        "pipeline_finalized",
        status=state.status,
        avg_confidence=round(state.avg_confidence, 3),
        claims=len(state.claims),
        total_latency_ms=total_latency,
    )
    return state.model_copy(update={
        "total_latency_ms": total_latency,
        "trace": state.trace + [{
            "step": "decision",
            "details": f"Finalized: status={state.status}, avg_confidence={state.avg_confidence:.3f}",
            "latency_ms": 0,
        }],
    })


# ── Retry retriever (increments retry_count, expands query) ──────────────────

async def retry_retriever_node(state: GraphState) -> GraphState:
    """
    On low-confidence retry: expand query with all sub_queries joined,
    bump retry_count, and re-retrieve.
    """
    expanded_query = f"{state.query} " + " ".join(state.sub_queries)
    expanded_state = state.model_copy(update={
        "query": expanded_query,
        "retry_count": state.retry_count + 1,
    })
    return await sub_query_retriever_node(expanded_state)


# ── Retry rewriter (increments retry_count) ──────────────────────────────────

async def retry_rewriter_node(state: GraphState) -> GraphState:
    """Rewrite low-quality claims and bump retry_count before re-verification."""
    bumped = state.model_copy(update={"retry_count": state.retry_count + 1})
    return await rewriter_node(bumped)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_verilayer_graph() -> StateGraph:
    """
    Construct and compile the VeriLayer LangGraph.
    Returns a compiled graph ready to invoke with GraphState.
    """
    builder = StateGraph(GraphState)

    # Register all nodes
    builder.add_node("planner",    planner_node)
    builder.add_node("retriever",  sub_query_retriever_node)
    builder.add_node("grader",     document_grader_node)
    builder.add_node("generator",  generator_node)
    builder.add_node("extractor",  claim_extractor_node)
    builder.add_node("verifier",   claim_verifier_node)
    builder.add_node("scorer",     confidence_scorer_node)
    builder.add_node("rewriter",   retry_rewriter_node)
    builder.add_node("retry_retriever", retry_retriever_node)
    builder.add_node("finalize",   finalize_node)

    # Linear pipeline edges
    builder.set_entry_point("planner")
    builder.add_edge("planner",   "retriever")
    builder.add_edge("retriever", "grader")
    builder.add_edge("grader",    "generator")
    builder.add_edge("generator", "extractor")
    builder.add_edge("extractor", "verifier")
    builder.add_edge("verifier",  "scorer")

    # Conditional routing from scorer via decision_node
    builder.add_conditional_edges(
        "scorer",
        decision_node,
        {
            ROUTE_FINALIZE: "finalize",
            ROUTE_REWRITE:  "rewriter",
            ROUTE_RETRIEVE: "retry_retriever",
        },
    )

    # After rewrite → re-verify (not re-extract)
    builder.add_edge("rewriter",        "verifier")

    # After retry retrieval → grade → generate → extract → verify
    builder.add_edge("retry_retriever", "grader")

    # Terminal
    builder.add_edge("finalize", END)

    return builder.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────

_graph = None


def get_verilayer_graph():
    """Return the cached compiled LangGraph instance."""
    global _graph
    if _graph is None:
        _graph = build_verilayer_graph()
        logger.info("verilayer_graph_compiled")
    return _graph


# ── Public pipeline runner ────────────────────────────────────────────────────

async def run_pipeline(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
) -> GraphState:
    """
    Run the full VeriLayer agentic pipeline for a query.

    Args:
        query: The user's question.
        top_k: Number of documents to retrieve per sub-query.
        session_id: Optional Langfuse session ID for trace grouping.

    Returns:
        Final GraphState with all fields populated.
    """
    graph = get_verilayer_graph()
    initial_state = GraphState(
        query=query,
        top_k=top_k,
        session_id=session_id,
        start_time_ms=int(time.time() * 1000),
    )
    logger.info("pipeline_starting", query=query[:80], top_k=top_k)
    final_state: GraphState = await graph.ainvoke(initial_state)
    logger.info(
        "pipeline_complete",
        status=final_state.status,
        avg_confidence=round(final_state.avg_confidence, 3),
        claims=len(final_state.claims),
        total_latency_ms=final_state.total_latency_ms,
    )
    return final_state
