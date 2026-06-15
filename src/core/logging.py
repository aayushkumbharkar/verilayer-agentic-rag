"""
VeriLayer — Structured JSON logging via structlog.
All logs are emitted as JSON for easy parsing by log aggregators.
"""
import logging
import sys
from typing import Any

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for structured JSON output.
    Call once at application startup inside the FastAPI lifespan.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "verilayer", **initial_values: Any) -> structlog.BoundLogger:
    """
    Get a named structured logger with optional bound context.

    Usage:
        logger = get_logger("ingestion", pipeline_id="abc123")
        logger.info("chunk created", chunk_id="xyz", token_count=256)
    """
    return structlog.get_logger(name).bind(**initial_values)


# Default application-level logger
logger: structlog.BoundLogger = get_logger("verilayer")
