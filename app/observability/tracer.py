from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import Counter

GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total",
    "Guardrail outcomes by stage and reason.",
    ["stage", "reason"],
)


def traced_span(name: str, attributes: dict[str, Any] | None = None):
    tracer = trace.get_tracer("retrieval-process-docs")
    try:
        context_manager = tracer.start_as_current_span(name)
        return _SpanContextManager(context_manager, attributes or {})
    except Exception:
        return nullcontext()


def mark_span_success(span: Span | None) -> None:
    if span is not None:
        span.set_status(Status(StatusCode.OK))


def mark_span_error(span: Span | None, exc: Exception) -> None:
    if span is not None:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))


class _SpanContextManager:
    def __init__(self, context_manager, attributes: dict[str, Any]):
        self._context_manager = context_manager
        self._attributes = attributes
        self._span: Span | None = None

    def __enter__(self) -> Span | None:
        self._span = self._context_manager.__enter__()
        if self._span is not None:
            for key, value in self._attributes.items():
                if value is not None:
                    self._span.set_attribute(key, value)
        return self._span

    def __exit__(self, exc_type, exc, tb):
        return self._context_manager.__exit__(exc_type, exc, tb)
