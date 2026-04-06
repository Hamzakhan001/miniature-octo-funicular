"""

RAGPipeline - fully async orchestration with parallel execution

Pipeline stages: 

Stage 1:  InputGuard + QueryClassifier run simultaneously
          Both are CPU-bound operations -> ThreadPool Executor
          Both are independed -> asyncio.gather


Stage 2: Hybrid Search( semantic -> BM25 re-rank)
         Sequential because BM25 needs semantic results


Stage 3: GPT-4o generation via AsyncOpenAI
         Pure async I/O - no thread needed


Stage 4: PII Check + Hallucination Detection simultaneous
         PII = CPU regex -> ThreadPool Executor
         Hallucination = CPU similarity -> ThreadPool Executor
         Both are independent -> asyncio.gather
        

Why ThreadPool Executor for CPU work:
Python's GIL mean only one thread runs python code at a time.
However, PyTorch(DisitlBert) and regex operations release GIL during computation. 
so threads genuinely runs in parallel for these.
ProcessPoolExecutor would bypass GIL entirely but has higher overhead and is overkill
for single-query processing.
"""


from __future__ import annotations
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.models import RAGResponse, GuardRailAction, DocumentChunk
from app.core.logging import log
from app.services.vector_store import VectorStoreService
from app.services.rag_chain import RAGServices
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from config.settings import get_settings


_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_pipeline")


_RAG_SYSTEM_PROMPT = (
    "You are helpful, precise assistant. Answer the user's question"
    "using Only the context proviede below. If the answer is not in the context"
    "context, say so clearly instead of guesstiong \n\nContext: \n {context}"

)


def _build_context(docs) -> str:
   return "/n/n".join(f"[source: {doc.metadata.get("source", "unkown")}] \n  {doc.page_content}" for doc in docs)


class RAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        self._vs = VectorStoreService()
        self._rag = RAGService(vector_store = self._vs)
        self._input_guard = InputGuard()
        self._output_guard = OutputGuard()

        try:
            from app.services.classifier import get_classifier
            self._classifier = get_classifier()
            self._has_classifier = True
        except Exception as e:
            self._classifier = None
            self._has_classifier = False
            log.warning("Classifier initialization failed, continuing without it")

    
    async def run(self, 
    question: str, 
    top_k: Optional[int] = None,
    filter: Optional[dict] = None,
    run_eval: bool = False,
    use_hybrid: bool = True
    ) -> RAGResponse:
    """Execute the full RAG pipeline asynchronously"""
    
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()

    if self._has_classifier:
        classify_task = loop.run_in_executor(_thread_pool, self._classifier.classify, question)
    else:
        async def allow_all():
            return {"should_proceed": True, "label":"answerable", "confidence": 1.0}
        classify_task = asyncio.ensure_future(allow_all())

    guard_task = loop.run_in_executor(_thread_pool, self._input_guard.check, question)

    classification, input_result = await asyncio.gather(classify_task, guard_task)

    log.info("stage 1 completed", classifier_label = classification.get("label", "skipped"),
    classifier_confidence = classification.get("confidence", 0.0),
    guard_action = input_result.action,
    elapsed_time = round((time.perf_counter() - t0)* 1000, 1))

    if not classification.get("should_proceed", True):
        return RAGResponse(
            answer = (
                f" Your question appears to be ourtside of scope of"
                f" available documents"
                f"(confidence: {classiciation.get("confidence", 0)})"
                f" Try rephrasing your question or ask about the available documentation."
            )
        )

    if input_result.action == GuardRailAction.BLOCK:
        return RAGResponse(
            answer = f"[BLOCKED] {input_result.reason}",
            guardrail = input_result,
            latency_ms = (time.perf_counter() - t0) *1000
        )

    
    effective_q = input_result.redacted_text or question

    if use_hybrid:
        docs = await self._vs.hybrib_search(
            effective_q,
            top_k = top_k or self._settings.top_k,
        )
        retrieval_method = "hybrid_bm25_semantic"
    else:
        docs = await self._vs.similarity_search(
            effective_q,
            top_k= top_k
        )
        retrieval_method = "semantic_only"

    log.info(
        "stage2_retrieval_completed",
        method=retrieval_method,
        docs_retrieved=len(docs)
        elapsed_time = round((time.perf_counter() - t0)*1000,1)

    )

    result = await self._rag.aquery(effective_q, docs=docs)
    answer = result["answer"]
    sources = result["sources"]
    
    log.info("stage3_generation_completed", answer_length=len(answer), sources= len(sources))

    chunks = [
        DocumentChunk(
            text=s["content"],
            metadata=s.get("metadata", {})
        )
        for s in sources
    ]

    output_result = await loop.run_in_executor(_thread_pool, lambda: self._output_guard.check(answer, chunks))

    if output_result.action == GuardRailAction.BLOCK:
        log.info("stage4_output_guard_blocked", reason=output_result.reason)
        return RAGResponse(
            answer = f"[BLOCKED] {output_result.reason}",
            guardrail = output_result,
            latency_ms = (time.perf_counter() - t0) *1000
        )

    
    final_answer = answer
    if output_result.action == GuardRailAction.REDACT:
        final_answer = output_result.redacted_text or answer
        log.info("stage4_output_guard_redacted", reason=output_result.reason)

    total_ms = (time.perf_counter() - t0) * 1000

    return RAGResponse(
        answer=final_answer,
        sources = sources,
        guardrail = output_result,
        latency_ms = total_ms
    )


async def _run_hallucination_check(self, answer: str, chunks: list):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _thread_pool,
            lambda: self._output_guard.check(answer, chunks)
        )
    except Exception as e:
        log.error("hallucination_check_failed", error=str(e))
        return False

    
