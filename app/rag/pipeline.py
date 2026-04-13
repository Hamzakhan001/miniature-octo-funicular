from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.observability.audit_writer import write_audit_record
from app.observability.audit import QueryAuditRecord
from app.core.logging import log
from app.core.models import DocumentChunk, GuardrailAction, RAGResponse
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.observability import (
    observe_answer_length,
    observe_eval_scores,
    observe_query_outcome,
    observe_retrieval,
    observe_stage_latency,
)
from app.observability.tracer import mark_span_error, mark_span_success, traced_span
from app.services.evaluation import EvaluationService
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService
from config.settings import get_settings

_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_pipeline")

_RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. "
    "Answer the user's question using only the provided context. "
    "If the context contains enough information, answer clearly and directly. "
    "If the context does not contain the answer, say that the answer is not available in the provided documents. "
    "Do not fabricate facts or rely on outside knowledge.\n\n"
    "Context:\n{context}"

)


def _build_context(docs) -> str:
    return "\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        self._vs = VectorStoreService()
        self._rag = RAGService(vector_store=self._vs)
        self._input_guard = InputGuard()
        self._output_guard = OutputGuard()
        self._evaluator = EvaluationService()

        try:
            from app.services.classifier import get_classifier

            self._classifier = get_classifier()
            self._has_classifier = True
        except Exception:
            self._classifier = None
            self._has_classifier = False
            log.warning("classifier_initialization_failed_continuing_without_it")

    async def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
        run_eval: bool = False,
        use_hybrid: bool = False,
    ) -> RAGResponse:
        stage_latencies_ms: dict[str, float] = {}
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        retrieval_method = "hybrid" if use_hybrid else "semantic"
        outcome = "success"

        with traced_span(
            "rag.query",
            {
                "rag.run_eval": run_eval,
                "rag.use_hybrid": use_hybrid,
                "rag.question_length": len(question),
            },
        ) as query_span:
            try:
                stage_start = time.perf_counter()
                if self._has_classifier:
                    classify_task = loop.run_in_executor(_thread_pool, self._classifier.classify, question)
                else:
                    async def allow_all():
                        return {"should_proceed": True, "label": "answerable", "confidence": 1.0}

                    classify_task = asyncio.create_task(allow_all())

                guard_task = loop.run_in_executor(_thread_pool, self._input_guard.check, question)
                classification, input_result = await asyncio.gather(classify_task, guard_task)
                observe_stage_latency("input_and_classification", time.perf_counter() - stage_start)

                log.info(
                    "stage1_completed",
                    classifier_label=classification.get("label", "skipped"),
                    classifier_confidence=classification.get("confidence", 0.0),
                    guard_action=input_result.action,
                    elapsed_time_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                stage_latencies_ms["input_and_classification"]=(time.perf_counter()-stage_start)*1000

                if not classification.get("should_proceed", True):
                    outcome = "out_of_scope"
                    response = RAGResponse(
                        answer=(
                            "Your question appears to be outside the scope of the available documents. "
                            f"(confidence: {classification.get('confidence', 0.0)})"
                        ),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                    mark_span_success(query_span)
                    return response

                if input_result.action == GuardrailAction.BLOCK:
                    outcome = "input_blocked"
                    response = RAGResponse(
                        answer=f"[BLOCKED] {input_result.reason}",
                        guardrail=input_result,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                    mark_span_success(query_span)
                    return response

                effective_question = input_result.redacted_text or question
                stage_start = time.perf_counter()
                with traced_span("rag.retrieval", {"rag.retrieval_method": retrieval_method}) as retrieval_span:
                    print("PIPELINE RETRIEVAL CONFIG:", {
                        "use_hybrid": use_hybrid,
                        "top_k": top_k,
                    })

                    if use_hybrid:
                        print("USING HYBRID SEARCH")
                        docs = await self._vs.hybrid_search(
                            query=effective_question,
                            top_k=top_k or self.settings.top_k,
                            filter=filter,
                        )
                    else:
                        print("USING SEMANTIC SEARCH")
                        docs = await self._vs.similarity_search(
                            query=effective_question,
                            top_k=top_k or self.settings.top_k,
                            filter=filter,
                        )
                    if retrieval_span is not None:
                        retrieval_span.set_attribute("rag.docs_retrieved", len(docs))
                    mark_span_success(retrieval_span)
                observe_stage_latency("retrieval", time.perf_counter() - stage_start)
                observe_retrieval(retrieval_method, len(docs))

                log.info(
                    "stage2_retrieval_completed",
                    method=retrieval_method,
                    docs_retrieved=len(docs),
                    elapsed_time_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                stage_latencies_ms["retrieval"]=(time.perf_counter()-stage_start)*1000

                stage_start = time.perf_counter()
                with traced_span("rag.generation", {"rag.docs_retrieved": len(docs)}) as generation_span:
                    result = await self._rag.aquery(effective_question, docs=docs)
                    answer = result["answer"]
                    sources = result["sources"]
                    if generation_span is not None:
                        generation_span.set_attribute("rag.answer_length", len(answer))
                    mark_span_success(generation_span)
                observe_stage_latency("generation", time.perf_counter() - stage_start)
                observe_answer_length(answer)
                log.info("stage3_generation_completed", answer_length=len(answer), sources=len(sources))
                stage_latencies_ms["generation"] = (time.perf_counter() - stage_start) * 1000

                chunks = [
                    DocumentChunk(text=source["content"], metadata=source.get("metadata", {}))
                    for source in sources
                ]
                stage_start = time.perf_counter()
                output_result = await loop.run_in_executor(
                    _thread_pool,
                    lambda: self._output_guard.check(answer, chunks),
                )
                observe_stage_latency("output_guard", time.perf_counter() - stage_start)

                if output_result.action == GuardrailAction.BLOCK:
                    outcome = "output_blocked"
                    log.info("stage4_output_guard_blocked", reason=output_result.reason)
                    response = RAGResponse(
                        answer=f"[BLOCKED] {output_result.reason}",
                        guardrail=output_result,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                    mark_span_success(query_span)
                    return response

                final_answer = (
                    output_result.redacted_text or answer
                    if output_result.action == GuardrailAction.REDACT
                    else answer
                )
                if output_result.action == GuardrailAction.REDACT:
                    outcome = "redacted"

                eval_scores = None
                if run_eval:
                    stage_start = time.perf_counter()
                    with traced_span("rag.evaluation") as eval_span:
                        eval_result = self._evaluator.evaluate(
                            question=effective_question,
                            answer=final_answer,
                            context=[source["content"] for source in sources],
                        )
                        eval_scores = eval_result.as_dict()
                        if eval_span is not None:
                            for metric, score in eval_scores.items():
                                eval_span.set_attribute(f"rag.eval.{metric}", score)
                        mark_span_success(eval_span)
                    observe_stage_latency("evaluation", time.perf_counter() - stage_start)
                    observe_eval_scores(eval_scores)
                    stage_latencies_ms["evaluation"] = (time.perf_counter() - stage_start) * 1000

                response = RAGResponse(
                    answer=final_answer,
                    sources=sources,
                    guardrail=output_result,
                    eval_scores=eval_scores,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                audit = QueryAuditRecord(
                question=question,
                top_k=top_k,
                retrieval_method=retrieval_method,
                docs_retrieved=len(sources),
                sources=[source.get("metadata", {}).get("source", "unknown") for source in sources],
                answer_length=len(final_answer),
                outcome=outcome,
                run_eval=run_eval,
                eval_scores=eval_scores,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                stage_latencies_ms=stage_latencies_ms,
                )
                write_audit_record(audit.model_dump())
                mark_span_success(query_span)
                return response
            except Exception as exc:
                outcome = "error"
                mark_span_error(query_span, exc)
                raise
            finally:
                observe_query_outcome(
                    retrieval_method=retrieval_method,
                    outcome=outcome,
                    run_eval=run_eval,
                    latency_seconds=time.perf_counter() - t0,
                )
