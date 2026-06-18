"""
VeriLayer — Agent Node: PlannerNode.
Decomposes the user query into focused sub-queries.
"""
from __future__ import annotations
import time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm_json
from src.rag.prompt_templates import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.planner")


@observe(name="planner-node")
async def planner_node(state: GraphState) -> GraphState:
    """Decompose the query into 2-4 sub-queries."""
    t0 = time.monotonic()
    update_span_metadata(input={"query": state.query})

    try:
        result = await call_llm_json(
            PLANNER_SYSTEM_PROMPT,
            PLANNER_USER_PROMPT.format(query=state.query),
        )
        sub_queries = result if isinstance(result, list) else [state.query]
        sub_queries = [str(q) for q in sub_queries[:4]]  # cap at 4
    except Exception as exc:
        logger.warning("planner_failed", error=str(exc))
        sub_queries = [state.query]

    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"sub_queries": sub_queries})
    logger.info("planner_complete", sub_queries=sub_queries, latency_ms=latency)

    return state.model_copy(update={
        "sub_queries": sub_queries,
        "trace": state.trace + [{"step": "planner", "details": f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}", "latency_ms": latency}],
    })
