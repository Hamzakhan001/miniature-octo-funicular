from __future__ import annotations

import asyncio
import os
import tempfile
from asyncio import Semaphore
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_store import VectorStoreService

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".html", ".csv"}


class IngestionService:
    """Async document ingestion service."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self._embedding_semaphore = Semaphore(10)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._node_parser = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    async def ingest_text(
        self,
        text: str,
        source: str = "manual",
        metadata: Optional[dict] = None,
    ) -> List[str]:
        chunks = await asyncio.to_thread(self._chunk_text_sync, text, source, metadata or {})
        ids = await self._batch_upsert(chunks)
        logger.info("ingested_text", source=source, count=len(ids))
        return ids

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: Optional[dict] = None,
    ) -> List[str]:
        chunks = await asyncio.to_thread(
            self._parse_and_chunk_sync,
            file_bytes,
            filename,
            metadata or {},
        )
        logger.info("file_parsed", filename=filename, chunks_before_upsert=len(chunks))
        ids = await self._batch_upsert(chunks)
        logger.info("file_ingested", filename=filename, count=len(ids))
        return ids

    async def ingest_directory(self, directory: str, metadata: Optional[dict] = None) -> List[str]:
        chunks = await asyncio.to_thread(self._load_directory_sync, directory, metadata or {})
        logger.info("directory_loaded", directory=directory, count=len(chunks))
        ids = await self._batch_upsert(chunks)
        logger.info("directory_ingested", directory=directory, count=len(ids))
        return ids

    async def _batch_upsert(self, chunks: List[Document], batch_size: int = 50) -> List[str]:
        if not chunks:
            return []

        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
        all_ids: List[str] = []
        logger.info("batch_upsert_start", total_batches=len(batches))

        for batch_num, batch in enumerate(batches, 1):
            async with self._embedding_semaphore:
                try:
                    ids = await self._upsert_batch_with_retry(batch, batch_num)
                    all_ids.extend(ids)
                    logger.info("batch_upsert_completed", batch_num=batch_num, count=len(ids))
                except Exception as exc:
                    logger.error("batch_upsert_error", batch_num=batch_num, error=str(exc))
                    raise
            await asyncio.sleep(0)

        return all_ids

    async def _upsert_batch_with_retry(
        self,
        batch: List[Document],
        batch_num: int,
        max_retries: int = 3,
    ) -> List[str]:
        for attempt in range(max_retries):
            try:
                return await self.vector_store.upsert_documents(batch)
            except Exception as exc:
                error_str = str(exc).lower()
                is_rate_limit = "rate limit" in error_str or "429" in error_str
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    raise

                wait_seconds = 5 * (2**attempt)
                if is_rate_limit:
                    logger.warning(
                        "rate_limit_hit",
                        batch_num=batch_num,
                        attempt=attempt + 1,
                        wait_seconds=wait_seconds,
                    )
                else:
                    logger.warning(
                        "upsert_error",
                        batch_num=batch_num,
                        attempt=attempt + 1,
                        wait_seconds=wait_seconds,
                    )
                await asyncio.sleep(wait_seconds)

        return []

    def _chunk_text_sync(self, text: str, source: str, metadata: dict) -> List[Document]:
        doc = Document(page_content=text, metadata={"source": source, **metadata})
        return self._splitter.split_documents([doc])

    def _parse_and_chunk_sync(self, file_bytes: bytes, filename: str, metadata: dict) -> List[Document]:
        suffix = Path(filename).suffix.lower()
        tmp_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            reader = SimpleDirectoryReader(input_files=[tmp_path], filename_as_id=True)
            llama_documents = reader.load_data()
            lc_docs = [
                Document(
                    page_content=doc.text,
                    metadata={**doc.metadata, "filename": filename, **metadata},
                )
                for doc in llama_documents
                if doc.text.strip()
            ]
            logger.info("file_chunked", filename=filename, count=len(lc_docs))
            return self._splitter.split_documents(lc_docs)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_directory_sync(self, directory: str, metadata: dict) -> List[Document]:
        reader = SimpleDirectoryReader(directory, recursive=True)
        llama_docs = reader.load_data()
        lc_docs = [
            Document(page_content=doc.text, metadata={**doc.metadata, **metadata})
            for doc in llama_docs
            if doc.text.strip()
        ]
        return self._splitter.split_documents(lc_docs)
