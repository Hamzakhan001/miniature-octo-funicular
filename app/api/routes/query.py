from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.models import DocumentChunk, GuardrailAction, QueryRequest, RAGResponse
from app.rag.pipeline import RAGPipeline, _RAG_SYSTEM_PROMPT, _build_context
from app.core.logging import log

router = APIRouter(prefix="/query", tags=["Query"])
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query_route")


_pipeline : RAGPipeline | None = None
_agent_graph = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline



def get_agent_graph():
    """
    Lazy singleton for the Langgraph agent.
    Reuses the same VectorStoreService and RAGService instances as linear pipeline - no duplication of pinecone connection
    """

    global _agent_graph
    if _agent_graph is None:
        from app.agents.graph import build_graph
        p=get_pipeline()
        _agent_graph = build_graph(p._vs, p._rag)
    return _agent_graph


@router.post("", response_model=RAGResponse, summary="Full RAG Query", description=(
    "Runs the complete async RAG pipeline: \n"
    "1. Input guard + query classiciation (happen in parallel)\n"
    "2. Hybrid search: semantic + BM25 reranking \n"
    "3. GPT-4O generation \n"
    "4. Output guard: PII scrub + hallucination check \n\n"
    "Set `run_eval=true` to include evaluation scores"
),
)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> RAGResponse:
    try:
        return await pipeline.run(
            question = body.question,
            top_k=body.top_k,
            filter=body.filter,
            run_eval=body.run_eval
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="Streaimg RAG query(sse)",
    description = (
        "Server sent events streaming endpoit \n"
        "Tokens are yielded as they are generated \n"
        "Input guard runs before streaming starts \n"
        "Output guardrails are not applied during streaming "
        "use the non-streaming endpoint for full guardrail validation"
    )
)
async def query_stream(body: QueryRequest):
    from app.guardrails.input_guard import InputGuard
    from config.settings import get_settings
    from openai import AsyncOpenAI

    settings = get_settings()
    loop = asyncio.get_event_loop()

    input_guard = InputGuard()
    check = await loop.run_in_executor(_thread_pool, input_guard.check, body.question)

    if check.action == GuardrailAction.BLOCK:
        async def blocked_stream():
            yield f"data: [BLOCKED] {check.reason} \n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    effective_q = check.redacted_text or body.question

    pipeline = get_pipeline()
    docs = await pipeline._vs.hybrid_search(effective_q, top_k=body.top_k or settings.top_k)
    content = _build_context(docs)

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def token_generator():
        try:
            stream = await client.chat.completions.create(
                model=settings.chat_model,
                temperature = settings.chat_model,
                messages = [
                    {
                        "role":"system",
                        "content": _RAG_SYSTEM_PROMPT.format(context=context)
                    },
                    {"role":"user", "content":effective_q}
                ],
                stream=True
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream", 
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.post("/agent", 
            response_model=RAGResponse, 
            summary="Agent RAG with guardrails",
            description = (
                "Langgraph-powered agent RAG with adaptive retrieval:\n\n"
                "1- **Input Guardrail** - injection detection, PII redaction, topic blocking\n"
                "2- **HybridRetrieval** - Semantic + keyword search with reranking\n"
                "3- **Relevance Grading** - LLM as a judge if docs answer the question\n"
                "4- **Query Rewrite** - query reformulation if docs are poor"
                "(up to 2 retries)\n"
                "5- **Generation** - LLM response with context\n"
                "6- **Output Guardrail** - safety and quality checks\n"
                "This endpoint will be used when retrival quality matters the most"
            )
)
async def agent_query(body: QueryRequest):
    """
    Agentic RAG via LangGraph

    Key differenc from query:
    Grades the retrieved docs for relevance to the query
    Rewrites the query and retries if docs are poor
    up to 2 rewrites before falling through to best effort generation
    """
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    from app.guardrails.input_guard import InputGuard
    from app.guardrails.output_guard import OutputGuard


    input_guard = InputGuard()
    output_guard = OutputGuard()

    check = await loop.run_in_executor(_thread_pool, input_guard.check, body.question)

    if check.action == GuardrailAction.BLOCK:
        raise HTTPException(status_code=400, detail=f"Input blocked: {check.reason}")


    effective_q = check.redacted_text or body.question

    graph = get_agent_graph()
    try:
        final_state = await graph.ainvoke({
            "question": effective_q,
            "rewritten_question": None,
            "docs": [],
            "answer": None,
            "sources": [],
            "grade": None,
            "retry_count": 0,
            "filter": body.filter,
            "top_k": body.top_k
        })
    except Exception as e:
        log.error("agent error", error=str(e), question=body.question[:80])
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    
    chunks = [
        DocumentChunk(
            text= s["content"], metadata = s.get("metadata", {})
        )
        for s in final_state["sources"]
    ]
    out_checks = await loop.run_in_executor(_thread_pool, lambda: output_guard.check(final_state["answer"], chunk, True))

    if out_checks.action == GuardrailAction.BLOCK:
        raise HTTPException(status_code=400, detail=f"Output blocked: {out_checks.reason}")

    answer = (
        out_checks.redacted_text if out_checks.action == GuardrailAction.REDACT else final_state["answer"]
    )

    return RAGResponse(
        answer=answer,
        sources = final_state["sources"],
        guardrail=out_check,
        latency_ms=(time.perf_counter() - t0) * 1000
    )

    
