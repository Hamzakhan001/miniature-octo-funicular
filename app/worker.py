"""
Celery worker - persistent background task queue unlike BackgroundTasks
Used for processing long running tasks like file ingestion
Persistent queue survive restarts, have expotential backoff,
retries, and can be monitored using Flower dashboard

Tasks: 
ingest single file into Pinecone reties upto 3 times with 60s backoff


Celery tasks are synchronous by default, they run in separate worker process.
We use asyncio.run() to run our async ingestion pipelien
Each task gets its own event loop in worker process
"""


from __future__ import annotations
import asyncio

from celery import Celery

from app.core.config import get_settings
from app.core.logging import setup_logging, logger


setup_logging()
settings = get_settings()

celery_app = Celery(
    "rag_woker",
    broker = settings.redis_url,
    backend = settings.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    task_track_started = True,
    result_expres = 86400,
    task_acks_late = True,
    worker_prefetch_multiplier = 1,

    task_soft_time_limit = 600,
    task_time_limit = 720,
)


@celery_app.task(
    bind= True,
    max_retries=3,
    default_retry_delay=60,
    name="rag_worker.ingest_file"
)
def ingest_file_task(self, file_bytes_hex: str, filename:str, metadata:dict) -> dict:
    try:
        logger.info(f"Ingesting file: {filename}")
        file_bytes = bytes.fromhex(file_bytes_hex)
        result = asyncio.run(_run_ingestion(file_bytes, filename, metadata))
        logger.info(f"Ingestion result: {result}")
        return result
    except Exception as e:
        logger.error("ingestion failed", filename=filename, task_id= self.request.id, attempts=seld.request.retries+1, max_retries=self.max_retries)
        retry_delay = 60*(2** self.request.retires)
        raise self.retry(
            exc = exc,
            countdown= retry_delay
        )


async def _run_ingestion(file_bytes: bytes, filename: str, metadata: dict) -> dict:

    from app.services.vector_store import VectorStoreService
    from app.services.ingestion import IngestionService
    
    vs = VectorStoreService()
    ingestion = IngestionService(vs)
    ids = await ingestion.ingest_file(file_bytes, filename, metadata)
    return {"status":"ok", "filename": filename, "chunks": len(ids), "ids":ids}
