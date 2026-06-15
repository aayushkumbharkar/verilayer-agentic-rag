"""
VeriLayer — Langfuse observability client.

Follows best practices from langfuse/skills SKILL.md:
- @observe decorator on every agent node
- Descriptive trace names (not 'trace-1')
- Model name + token usage captured on generations
- Proper span hierarchy: trace → span → generation
- langfuse.flush() called at shutdown
- langfuse_context used for mid-function metadata updates
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe  # noqa: F401  (re-exported)

from src.core.config import settings

logger = structlog.get_logger("verilayer.langfuse")


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """
    Return a cached Langfuse client, or None if keys are not configured.
    Gracefully degrades so the application works without Langfuse.
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("langfuse_keys_missing", msg="Langfuse tracing disabled — keys not set")
        return None

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    logger.info("langfuse_client_initialized", host=settings.langfuse_host)
    return client


def flush_langfuse() -> None:
    """Flush all pending Langfuse events — call at application shutdown."""
    client = get_langfuse_client()
    if client:
        client.flush()
        logger.info("langfuse_flushed")


def update_trace_metadata(
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Update the current trace's metadata from within an @observe-decorated function.
    Safe to call — no-ops if no active trace.
    """
    try:
        kwargs: dict[str, Any] = {}
        if name:
            kwargs["name"] = name
        if user_id:
            kwargs["user_id"] = user_id
        if session_id:
            kwargs["session_id"] = session_id
        if tags:
            kwargs["tags"] = tags
        if metadata:
            kwargs["metadata"] = metadata
        if kwargs:
            langfuse_context.update_current_trace(**kwargs)
    except Exception:
        pass  # Tracing must never break the application


def update_span_metadata(
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
) -> None:
    """
    Update the current span's input/output from within an @observe-decorated function.
    Safe to call — no-ops if no active span.
    """
    try:
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata
        if level:
            kwargs["level"] = level
        if kwargs:
            langfuse_context.update_current_observation(**kwargs)
    except Exception:
        pass


def update_generation_metadata(
    model: str,
    input_tokens: int,
    output_tokens: int,
    model_parameters: dict[str, Any] | None = None,
) -> None:
    """
    Update token usage and model info for an LLM generation span.
    Must be called inside an @observe(as_type='generation') decorated function.
    """
    try:
        langfuse_context.update_current_observation(
            model=model,
            usage={"input": input_tokens, "output": output_tokens},
            model_parameters=model_parameters or {},
        )
    except Exception:
        pass
