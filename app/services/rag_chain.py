from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional

from langchain_core.documents import Document
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_store import VectorStoreService


class RAGService:
    """Minimal async RAG service used by the API pipeline."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self._client = AsyncOpenAI(api_key=self.settings.open_ai_api_key)

    async def astream(self, question: str) -> AsyncGenerator[str, None]:
        result = await self.aquery(question=question)
        yield result["answer"]

    async def aquery(
        self,
        question: str,
        docs: Optional[List[Document]] = None,
        filter: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        from app.rag.pipeline import _build_context

        if docs is None:
            docs = await self.vector_store.similarity_search(
                query=question,
                top_k=top_k or self.settings.top_k,
                filter=filter,
            )

        context = _build_context(docs)
        response = await self._client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, precise assistant. Answer using ONLY the "
                        "context below. If the answer is not in the context, say so clearly.\n\n"
                        f"Context:\n{context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        logger.info("rag_query", question=question[:80], sources=len(docs))
        return {
            "answer": answer,
            "sources": [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in docs
            ],
        }
