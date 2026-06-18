"""
VeriLayer — Phase 5: Groq LLM generator with retry + timeout.
All LLM calls go through this module — never call Groq directly in agent nodes.
"""
from __future__ import annotations

import json
import re
import structlog
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.observability.langfuse_client import observe, update_generation_metadata

logger = structlog.get_logger("verilayer.rag.generator")


def _get_groq_client() -> AsyncGroq:
    return AsyncGroq(api_key=settings.groq_api_key)


@observe(name="groq-chat-completion", as_type="generation")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Call Groq LLM with retry and timeout.
    Decorated with @observe for Langfuse tracing — captures model + token usage.
    """
    client = _get_groq_client()
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.groq_timeout,
    )

    content = response.choices[0].message.content or ""
    usage = response.usage

    # Log token usage to Langfuse per SKILL.md best practices
    update_generation_metadata(
        model=settings.groq_model,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
    )

    logger.info(
        "llm_call_complete",
        model=settings.groq_model,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
    return content


async def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> dict | list:
    """
    Call LLM and parse JSON response. Strips markdown code fences if present.
    Raises ValueError if response is not valid JSON.
    """
    raw = await call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=1024)
    # Strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("llm_json_parse_failed", raw=raw[:200], error=str(exc))
        raise ValueError(f"LLM returned non-JSON: {raw[:200]}") from exc
