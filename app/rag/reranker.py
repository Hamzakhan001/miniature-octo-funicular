"""Simple reranker stub - returns docs as-is (no Cohere key required)""""


from __future__ import annotations
from typing import List
from langchain_core.documents import Document


class Reranker:
    def rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        return docs[:top_n]