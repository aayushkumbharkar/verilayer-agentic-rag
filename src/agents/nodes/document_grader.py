"""
VeriLayer — Agent Node: DocumentGrader.
Filters retrieved docs to keep only relevant ones.
"""
from __future__ import annotations
import asyncio, time
import structlog
from src.agents.state import GraphState
from src.observability.langfuse_client import observe, update_span_metadata
from src.rag.generator import call_llm
from src.rag.prompt_templates import GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT

logger = structlog.get_logger("verilayer.agents.grader")


@observe(name="document-grader")
async def document_grader_node(state: GraphState) -> GraphState:
    """Grade each retrieved doc as relevant or irrelevant."""
    t0 = time.monotonic()

    if not state.retrieved_docs:
        return state.model_copy(update={"graded_docs": [], "trace": state.trace + [
            {"step": "grader", "details": "No documents to grade", "latency_ms": 0}]})

    async def grade_doc(doc: dict) -> dict | None:
        try:
            verdict = await call_llm(
                GRADER_SYSTEM_PROMPT,
                GRADER_USER_PROMPT.format(query=state.query, chunk_text=doc["text"][:600]),
                temperature=0.0, max_tokens=16,
            )
            return doc if "relevant" in verdict.lower() else None
        except Exception:
            return doc  # keep on error

    results = await asyncio.gather(*[grade_doc(d) for d in state.retrieved_docs])
    graded = [d for d in results if d is not None]

    latency = int((time.monotonic() - t0) * 1000)
    update_span_metadata(output={"total": len(state.retrieved_docs), "kept": len(graded)})
    logger.info("grading_complete", total=len(state.retrieved_docs), kept=len(graded))

    return state.model_copy(update={
        "graded_docs": graded,
        "trace": state.trace + [{"step": "grader", "details": f"Kept {len(graded)}/{len(state.retrieved_docs)} relevant docs", "latency_ms": latency}],
    })
