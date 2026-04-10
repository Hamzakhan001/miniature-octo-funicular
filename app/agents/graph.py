"""
LangGraph agent graph - adaptive RAG with query rewritting

Flow:
retrive -> grade_docs -> [relevant] -> generate -> END
                      -> [irrelevant + retries left] -> rewrite_query -> retrieve(loop) 
                      -> [irrelevant + exhausted] -> generate (best effort)


MAX_RETRIES = 2 means the question can be rewritten at most 2 times
we fall through to generation anway - prevents infine loops
"""

from __future__ import annotations
from functools import partial
from langgraph.graph import END, StateGraph

from app.agents.nodes import generate, grade_documents, retrieve, rewrite_query
from app.agents.state import AgentState


MAX_RETRIES = 2


def _route_after_grade(state: AgentState) -> str:
    if state["grade"] == "relevant":
        return "generate"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "generate"


def build_graph(vs, rag):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", partial(retrieve, vs=vs))
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", partial(generate, rag=rag))


    graph.set_entry_point("retrieve")
    graph.add_node("grade_docs", grade_docs)
    graph.add_conditional_edges(
        "grade_docs", _route_after_grade
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    
    return graph.compile()

    


