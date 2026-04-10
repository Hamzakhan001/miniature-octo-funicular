from __future__ import annotations
from openai import AsyncOpenAI
from app.core.logging import logger

asyc def retrieve(state: AgentState, vs) -> AgentState:
    q = state.get("rewritten_question") or state["question"]
    top_k = state.get("top_k") or 5
    docs = await vs.hybrid_search(q, top_k=top_k)
    logger.info("agent_retrieve", query=q[:60], docs=len(docs))
    return {**state, "docs": docs}


async def grade_docs(state: AgentState) -> AgentState:
    q = state.get("rewritten_question") or state["question"]
    top_k = state.get("top_k") or 5
    docs = await vs.hybrid_search(q, top_k=top_k)
    logger.info("agent_retrieve", query=q[:60], docs=len(docs))
    return {**state, "docs": docs}
    
async def grad_docs(state: AgentState) -> AgentState:
    """
    """
    if not state["docs"]:
        return {**state, "grade": "irrelevant"}

    settings = get_settings()
    context = "\n --- \n".join(d.page_content[:300] for d in state["docs"][:3])
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        response = await client.chat.completions.create(
            model = settings.openai_chat_model,
            temperature=0.0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": "Respond with ONLY 'relevant' or 'irrelevant",
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state["question"]} \n\n "
                        f"Documents: \n{Context}"
                    ),
                }
            ]
        )
        raw = response.choices[0].message.content.strip().lower()
        grade = "relevant" if "relevant" in raw else "irrelevant"
    except Exception as e:
        logger.error("agent_grade_failed", grade=grade, question=state["question"][:60])
        grade = "relevant"
    
    return {**state, "grade": grade}


async def rewrite_query(state: AgentState) -> AgentState:
    settings = get_settings()

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        response = await client.chat.completions.create(
            model = settings.openai_chat_model,
            temperature=0.3,
            max_tokens=100,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the question to be more specific and retrival friendly"
                        "Output ONLY the rewritten question, nothing else"
                    ),
                },
                {
                    "role":"user",
                    "content": state["question"]
                }
            ]
        )

        rewritten = response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("agent_rewrite_failed", error=str(e))
        rewritten = state["question"]
    
    return {
        **state,
        "rewritten_question": rewritten,
        "rety_count": state.get("retry_count", 0)+1
    }


async def generate(state: AgentState, rag) -> AgentState:
    question = state.get("rewritten_question") or state["question"]
    result = await rag.aquery(question=question, docs=state["docs"])
    return {**state, "answer": result["answer"], "sources": result["sources"]}

