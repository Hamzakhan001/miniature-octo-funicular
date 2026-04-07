from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import logger

_thread_pool = ThreadPoolExcecutor(max_workers=4, thread_name_prefix="pinecone")


class VectorStoreService:
    def __init__(self):
        self._pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self._async_openai = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self._embeddings = OpenAIEmbeddings(
            model = self.settings.openai_embedding_model,
            openai_api_key= self.settings.openai_api_key
        )

        self._store: Optional[PineconeVectorStore] = None

        self._ensure_index()

    
    def _ensure_index(self) -> None:
        existing_names = [idx.name for idx in self._pc.list_indexes()]

        if self.settings.pinecone_index_name not in existing_names:
            logger.info(
                "creating piplines index",
                name = self.settings.pinecone_index_name,
                dimension = self.settings._embedding_semaphore,
            ),

            self._pc.create_index(
                name=self.settings.pinecone_index_name,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment,
                ),
                dimension=self.settings._embedding_semaphore,
                metric="cosine"
            )

            while not self._pc.describe_index(
                self.settings.pinecone_index_name
            ).status["ready"]:
                time.sleep(1)
            logger.info("index is ready")
        else:
            logger.info("index already exists")

        self._store = PineconeVectorStore(
            index_name=self.settings.pinecone_index_name,
            embedding=self._embeddings,
            pinecone_api_key=self.settings.pinecone_api_key,
        )

        logger.info("vector store initialized")


    async def upsert_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        embeddings = await self._embed_texts_async(texts)

        loop = asyncio.get_event_loop()
        ids = await loop.run_in_executor(
            _thread_pool,
            lamda: self._pinecone_upsert_sync(texts,embeddings, metadatas)
        )

        logger.info("documents upserted", ids=ids)
        return ids

    
    async def similarity_search(self, query: str, top_k: Optional[int] = None,
                                filer: Optional[dict] = None) -> List[Document]:
        loop = asyncio.get_event_loop()
        semantic_docs = await loop.run_in_executor(
            _thread_pool,
            lambda: self._store.similarity_search(query, k=top_k, filter=filer)
        )

        if not semantic_docs:
            return []

        reranked = await loop.run_in_executor(
            _thread_pool,
            lambda: self._bm25_rerank_sync(query, semantic_docs, top_k, alpha)
        )

        logger.info("hybrid search complete")

        return reranked

    
    async def delete_docuements(self, ids: List[str]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _thread_pool,
            lambda: self._delete_sync(ids)
        )

        logger.info("documents deleted", ids=ids)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        kwargs = search_kwargs or {"k": self.settings.top_k}
        return self._store.as_retriever(search_kwargs=kwargs)
        

    async def _embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        response = await self._async_openai.embeddings.create(
        model = self.settings.openai_embedding_model,
        input=texts
        )

        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item..embedding for item in sorted_data]


    def _similarity_upsert_sync(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> List[str]:
        index = self._pc.Index(self.settings.pinecone_index_name)
        ids = [str(uuid.uuid4()) for _ in texts]

        vectors = [
            {
                "id": ids[i],
                "values": embeddings[i],
                "metadatas": {
                    **metadatas[i],
                    "text": texts[i]
                }
            }
            for i in range(len(texts))
        ]
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors = vectors[i: i+batch_size])
        
        return ids

    
    def _bm25_rerank_sync(
        self,
        query: str,
        docs: List[Document],
        top_k: int,
        alpha: float
    ) -> List[Document]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank-bm25: pip install rank-bm25")
            return docs[:top_k]


        if not docs:
            return []

        tokenized_corpus = [doc.page_conten.lower().split() for doc  in docs]
        bm25 = BM250kapi(tokenized_corpus)
        bm25_score = bm25.get_scores(query.lower().split())

        k=60
        n=len(docs)

        semantic_rrf = [1/(k+rank+1) for rank in range(n)]

        bm25_ranked = sorted(
            range(n), key=lambda i: bm25_score[i], reverse=True
        )

        bm25_rank_map = {doc_idx: rank for rank,doc_idx in enumerate(bm25_ranked)}
        bm25_rrf = [1/(k+bm25_rank_map[i]+1) for i in range(n)]

        combined_scores = [
            alpha*sem+1(1-alpha) *bm25
            for sem,bm25 in zip(semantic_rrf, bm25_rrf)
        ]

        sorted_pairs = sorted(
            zip(docs, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in sorted_pairs[:top_k]]

    
    def _delete_sync(self, ids: List[str]) ->None:
        index = self._pc.Index(self.settings.pinecone_index_name)
        index.delete(ids=ids)