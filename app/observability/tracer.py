from __future__ import annotations

from contextlib import nullcontext

from opentelemetry import trace
from prometheus_client import Counter

GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total",
    "Guardrail outcomes by stage and reason.",
    ["stage", "reason"],
)


def traced_span(name: str):
    tracer = trace.get_tracer("retrieval-process-docs")
    try:
        return tracer.start_as_current_span(name)
    except Exception:
        return nullcontext()
