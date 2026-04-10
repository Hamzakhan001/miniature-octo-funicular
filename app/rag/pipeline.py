from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.logging import log
from app.core.models import DocumentChunk, GuardrailAction, RAGResponse
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.services.evaluation import EvaluationService
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService
from config.settings import get_settings

_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_pipeline")

_RAG_SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. Answer the user's question using only "
    "the provided context. If the answer is not in the context, say so clearly.\n\n"
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
        use_hybrid: bool = True,
    ) -> RAGResponse:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        if self._has_classifier:
            classify_task = loop.run_in_executor(_thread_pool, self._classifier.classify, question)
        else:
            async def allow_all():
                return {"should_proceed": True, "label": "answerable", "confidence": 1.0}

            classify_task = asyncio.create_task(allow_all())

        guard_task = loop.run_in_executor(_thread_pool, self._input_guard.check, question)
        classification, input_result = await asyncio.gather(classify_task, guard_task)

        log.info(
            "stage1_completed",
            classifier_label=classification.get("label", "skipped"),
            classifier_confidence=classification.get("confidence", 0.0),
            guard_action=input_result.action,
            elapsed_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

        if not classification.get("should_proceed", True):
            return RAGResponse(
                answer=(
                    "Your question appears to be outside the scope of the available documents. "
                    f"(confidence: {classification.get('confidence', 0.0)})"
                ),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if input_result.action == GuardrailAction.BLOCK:
            return RAGResponse(
                answer=f"[BLOCKED] {input_result.reason}",
                guardrail=input_result,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        effective_question = input_result.redacted_text or question
        if use_hybrid:
            docs = await self._vs.hybrid_search(
                query=effective_question,
                top_k=top_k or self.settings.top_k,
                filter=filter,
            )
            retrieval_method = "hybrid"
        else:
            docs = await self._vs.similarity_search(
                query=effective_question,
                top_k=top_k or self.settings.top_k,
                filter=filter,
            )
            retrieval_method = "semantic"

        log.info(
            "stage2_retrieval_completed",
            method=retrieval_method,
            docs_retrieved=len(docs),
            elapsed_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

        result = await self._rag.aquery(effective_question, docs=docs)
        answer = result["answer"]
        sources = result["sources"]
        log.info("stage3_generation_completed", answer_length=len(answer), sources=len(sources))

        chunks = [
            DocumentChunk(text=source["content"], metadata=source.get("metadata", {}))
            for source in sources
        ]
        output_result = await loop.run_in_executor(
            _thread_pool,
            lambda: self._output_guard.check(answer, chunks),
        )

        if output_result.action == GuardrailAction.BLOCK:
            log.info("stage4_output_guard_blocked", reason=output_result.reason)
            return RAGResponse(
                answer=f"[BLOCKED] {output_result.reason}",
                guardrail=output_result,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        final_answer = output_result.redacted_text or answer if output_result.action == GuardrailAction.REDACT else answer

        eval_scores = None
        if run_eval:
            eval_result = self._evaluator.evaluate(
                question=effective_question,
                answer=final_answer,
                context=[source["content"] for source in sources],
            )
            eval_scores = eval_result.as_dict()

        return RAGResponse(
            answer=final_answer,
            sources=sources,
            guardrail=output_result,
            eval_scores=eval_scores,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
