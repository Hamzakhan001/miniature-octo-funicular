from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

HTTP_INSTRUMENTATION = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"],
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries processed.",
    ["retrieval_method", "outcome", "run_eval"],
)

RAG_QUERY_LATENCY_SECONDS = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency.",
    ["retrieval_method", "outcome"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)

RAG_STAGE_LATENCY_SECONDS = Histogram(
    "rag_stage_latency_seconds",
    "Latency by RAG pipeline stage.",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

RAG_RETRIEVED_DOCS = Histogram(
    "rag_retrieved_docs",
    "Number of documents retrieved per query.",
    ["retrieval_method"],
    buckets=(0, 1, 2, 3, 5, 8, 10, 20, 50),
)

RAG_ANSWER_LENGTH = Histogram(
    "rag_answer_length_chars",
    "Generated answer length in characters.",
    buckets=(0, 50, 100, 250, 500, 1000, 2000, 4000, 8000),
)

RAG_EVAL_SCORE = Histogram(
    "rag_eval_score",
    "Evaluation score distribution by metric.",
    ["metric"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def configure_metrics(app: FastAPI) -> None:
    HTTP_INSTRUMENTATION.instrument(app).expose(
        app,
        include_in_schema=False,
        endpoint="/metrics",
        tags=["Observability"],
    )


def observe_stage_latency(stage: str, seconds: float) -> None:
    RAG_STAGE_LATENCY_SECONDS.labels(stage=stage).observe(seconds)


def observe_retrieval(method: str, docs_count: int) -> None:
    RAG_RETRIEVED_DOCS.labels(retrieval_method=method).observe(docs_count)


def observe_answer_length(answer: str) -> None:
    RAG_ANSWER_LENGTH.observe(len(answer))


def observe_eval_scores(scores: dict[str, float] | None) -> None:
    if not scores:
        return
    for metric, score in scores.items():
        RAG_EVAL_SCORE.labels(metric=metric).observe(score)


def observe_query_outcome(
    *,
    retrieval_method: str,
    outcome: str,
    run_eval: bool,
    latency_seconds: float,
) -> None:
    run_eval_label = "true" if run_eval else "false"
    RAG_QUERIES_TOTAL.labels(
        retrieval_method=retrieval_method,
        outcome=outcome,
        run_eval=run_eval_label,
    ).inc()
    RAG_QUERY_LATENCY_SECONDS.labels(
        retrieval_method=retrieval_method,
        outcome=outcome,
    ).observe(latency_seconds)
