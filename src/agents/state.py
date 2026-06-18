"""
VeriLayer — Phase 6: LangGraph GraphState.
The single shared state object that flows through all agent nodes.
"""
from __future__ import annotations
from typing import Annotated, Any
from pydantic import BaseModel, Field
import operator


class GraphState(BaseModel):
    """
    Shared state for the VeriLayer LangGraph pipeline.
    All nodes read from and write to this object.
    """
    # Input
    query: str = Field(description="Original user query")
    top_k: int = Field(default=5)
    session_id: str | None = Field(default=None)

    # Planner output
    sub_queries: list[str] = Field(default_factory=list)

    # Retrieval output
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    graded_docs: list[dict[str, Any]] = Field(default_factory=list)

    # Generation
    answer: str = Field(default="")

    # Claim extraction + verification
    claims: list[dict[str, Any]] = Field(default_factory=list)

    # Confidence + status
    avg_confidence: float = Field(default=0.0)
    status: str = Field(default="unsafe")  # verified | partial | unsafe

    # Trace
    trace: list[dict[str, Any]] = Field(default_factory=list)

    # Retry management
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=2)

    # Timing
    start_time_ms: int = Field(default=0)
    total_latency_ms: int = Field(default=0)

    class Config:
        arbitrary_types_allowed = True
