"""
VeriLayer — Agent Node: GeneratorNode.
Generates the final answer from graded documents.
"""
from __future__ import annotations
import time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm
from src.rag.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.generator")


@observe(name="generator-node")
async def generator_node(state: GraphState) -> GraphState:
    """Generate answer from graded documents. Falls back if no docs."""
    t0 = time.monotonic()
    docs = state.graded_docs or state.retrieved_docs

    if not docs:
        return state.model_copy(update={
            "answer": "Insufficient source documents to answer this query.",
            "status": "unsafe",
            "trace": state.trace + [{"step": "generator", "details": "No documents available — returned fallback", "latency_ms": 0}],
        })

    context = "\n\n---\n\n".join(
        f"[Source: {d.get('source','?')} | Section: {d.get('section','?')}]\n{d['text']}"
        for d in docs[:settings_top_k(state)]
    )
    update_span_metadata(input={"query": state.query, "doc_count": len(docs)})

    answer = await call_llm(
        RAG_SYSTEM_PROMPT,
        RAG_USER_PROMPT.format(context=context, query=state.query),
        temperature=0.1, max_tokens=2048,
    )
    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"answer_preview": answer[:200]})
    logger.info("generation_complete", answer_len=len(answer), latency_ms=latency)

    return state.model_copy(update={
        "answer": answer,
        "trace": state.trace + [{"step": "generator", "details": f"Generated {len(answer)}-char answer from {len(docs)} docs", "latency_ms": latency}],
    })


def settings_top_k(state: GraphState) -> int:
    return min(state.top_k, len(state.graded_docs or state.retrieved_docs))
