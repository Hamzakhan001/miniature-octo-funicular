from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document

class AgentState(TypeDict):
    question: str
    rewritten_question: Optional[str]
    docs: List[Document]
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    grade: Optional[str]
    retry_count: int
    filter: Optional[Dict[str, Any]]
    top_k: Optional[int]