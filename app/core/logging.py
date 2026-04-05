from __future__ import annotations
import logging
import sys
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars
from config.settings import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)


def bind_request_context(request_id: str, trace_id: str = "") -> None:
    bind_contextvars(request_id=request_id, trace_id=trace_id)


def clear_request_context() -> None:
    clear_contextvars()


log = structlog.get_logger()
logger = log