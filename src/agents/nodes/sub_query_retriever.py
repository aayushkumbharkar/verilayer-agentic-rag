"""
VeriLayer — Agent Node: SubQueryRetriever.
Retrieves documents for each sub-query using hybrid search.
"""
from __future__ import annotations
import asyncio
import time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.models.schemas import SearchRequest
from src.retrieval.hybrid import hybrid_search

logger = structlog.get_logger("verilayer.agents.retriever")


@observe(name="sub-query-retriever")
async def sub_query_retriever_node(state: GraphState) -> GraphState:
    """Run hybrid search for each sub-query, deduplicate by chunk_id."""
    t0 = time.monotonic()
    update_span_metadata(input={"sub_queries": state.sub_queries, "top_k": state.top_k})

    all_docs: dict[str, dict] = {}

    async def retrieve_one(query: str):
        try:
            req = SearchRequest(query=query, top_k=state.top_k)
            resp = await hybrid_search(req)
            for r in resp.results:
                if r.chunk_id not in all_docs:
                    all_docs[r.chunk_id] = r.model_dump()
        except Exception as exc:
            logger.warning("retrieval_failed", query=query, error=str(exc))

    await asyncio.gather(*[retrieve_one(q) for q in state.sub_queries])
    docs = list(all_docs.values())

    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"retrieved_count": len(docs)})
    logger.info("retrieval_complete", docs=len(docs), latency_ms=latency)

    return state.model_copy(update={
        "retrieved_docs": docs,
        "trace": state.trace + [{"step": "retriever", "details": f"Retrieved {len(docs)} unique chunks from {len(state.sub_queries)} sub-queries", "latency_ms": latency}],
    })
