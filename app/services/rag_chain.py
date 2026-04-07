from __future__ import annotations
from typing import List, Optional, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langhcain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document


from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_store import VectorStoreService


_RAG_SYSTEM_PROMPT = """
You are helpful, precise assistant. Answer the user's question \
using the provided context. If the context doesn't contain enough information, \
say so clearly.

context:
{context}

"""

RAG_HUMAN_PROMPT = "{question}"


def _format_docs(docs: List[Document]) -> str:
    return "\\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n {d.page_content}"
        for d in docs
    )


class RAGService:
    """LangChain LCEL-based RAG chain with streaming support"""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self._llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            openai_api_key=self.settings.openai_api_key,
            temperature= 0.2,
            streaming = True
        )

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT)
        ])
        self._chain = self._build_chain()
    
    def _build_chain(self):
        retriever = self.vector_store.as_retriever()
        setup = RunnableParallel({
            "context": self._retrieve_context,
            "question": RunnablePassthrough()
        }) 
        
        return setup | self._prompt | self._llm | StrOutputParser()

    def query(
        self,
        question: str,
        filter: Optional[dict] = None,
        top_k: Optional[dict] = None,
    ) -> dict:
        docs = self.vector_store.similarity_search(question, top_k=top_k, filter=filter)
        context = _format_docs(docs)
        prompt_value = self._prompt.format_messages(context=context, question=question)
        llm_sync = ChatOpenAI(
            model = self.settings.openai_chat_model,
            openai_api_key = self.settings.openai_api_key,
            temperature = 0.2
        )
        response = llm_sync.invoke(prompt_value).content
        logger.info("rag_query", question=question[:80], sources=len(docs))
        return {
            "answer": answer,
            "sources": [
                {
                    "content": d.page_content[:300],
                    "metadata": d.metadata
                }
                for d in docs
            ],
        }

    async def astream(self, question: str) -> AsyncGenerator[str, None]:
        async for chunk in self._chain.astream(question):
            yield chunk

    async def aquery(
        self,
        question: str,
        docs: list = None,
        filter: dict = None,
        top_k: int = None,
    ) -> dict:
        from openai import AsyncOpenAI
        from app.rag.pipeline import _build_context

        settings = self.settings

        if docs is None:
            docs = await asyncio.get_event_loop().run_in_executor(None, lambda: self.vector_store._similarity_search_sync(
                question, top_k or settings.top_k, filter)
            )

        context = _build_context(docs)
        client = AsyncOpenAI(api_key = settings.openai_api_key)
        response = await client.chat.completions.create(
            model = settings.openai_chat_model,
            messages = [
                {"role": "system", "content": "content": (
                        "You are a helpful, precise assistant. Answer using "
                        "ONLY the context below. If the answer is not in the "
                        "context, say so clearly.\n\nContext:\n" + context
                    ),
                },
                {"role": "user", "content": question}
            ],
            temperature = 0.2
        )
        answer = response.choices[0].message.content or ""
        logger.info("rag_query", question=question[:80], sources=len(docs))
        return {
            "answer": answer,
            "sources": [
                {
                    "content": d.page_content[:300],
                    "metadata": d.metadata
                }
                for d in docs
            ],
        }


        def query_with_rerank(
            self,
            quesiton: str,
            top_k: int=10,
            rerank_top_n: int=3,
        ) -> dict:
            try:
                from llama_index.core import VectorStoreIndex
                from llama_index.postprocessor.cohere_rerank import CohereRerank
            except ImportError:
                logger.warning("Cohere Rerank not available. Install llama-index-postprocessor-cohere-rerank")
                return self.query(question, top_k = rerank_top_n)

            docs = self.vector_store.similarity_search(question, k = top_k)
            docs=docs[:rerank_top_n]
            context = _format_docs(docs)
            prompt_value=self._prompt.format_messages(context=context, question=question)
            llm_sync = ChatOpenAI(
                model=self.settings.openai_chat_model,
                openai_api_key= self.settings.openai_api_key,
                temperature=0.2
            )
            answer = llm_sync.invoke(prompt_value).content
            return {"answer": answer, "sourcs":[{"content": d.page_content[:300], metadata: d.metadata} for d in docs]}


            
