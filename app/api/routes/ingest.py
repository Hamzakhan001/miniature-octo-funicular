from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.models import DeleteRequest, IngestionResult, IngestionTextRequest

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

MAX_FILE_SIZE = 50 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".html"}
SYNC_THRESHOLD_BYTES = 1 * 1024 * 1024


class AsyncIngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str
    check_status_url: str


class TaskStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def get_ingestion():
    from app.api.deps import get_ingestion_service

    return get_ingestion_service()


@router.post(
    "/text",
    response_model=IngestionResult,
    summary="Ingest raw text",
    description="Ingest raw text directly into the vector store.",
)
async def ingest_text(body: IngestionTextRequest):
    try:
        ingestion = get_ingestion()
        ids = await ingestion.ingest_text(body.text, body.metadata, source=body.source)
        return IngestionResult(status="ok", chunks=len(ids), ids=ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/file",
    summary="Ingest a file",
    description="Upload and ingest a file asynchronously via celery.",
)
async def ingest_file(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {SUPPORTED_EXTENSIONS}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB",
            )

        filename = file.filename or "upload"
        if len(content) < SYNC_THRESHOLD_BYTES:
            ingestion = get_ingestion()
            ids = await ingestion.ingest_file(file.file, filename)
            return IngestionResult(status="ok", chunks=len(ids), ids=ids)

        job_id = f"job_{int(time.time())}"
        try:
            from app.worker import ingest_file_task

            task = ingest_file_task.delay(
                content.hex(),
                filename,
                {"source": filename, "queued_at": time.time()},
            )
            return AsyncIngestResponse(
                job_id=task.id if getattr(task, "id", None) else job_id,
                status="queued",
                filename=filename,
                message="File queued for processing.",
                check_status_url=f"/api/v1/ingest/status/{job_id}",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/status/{job_id}",
    response_model=TaskStatusResponse,
    summary="Check ingestion job status",
    description="Poll this endpoint for Celery task state.",
)
def get_task_status(job_id: str):
    try:
        from celery.result import AsyncResult

        from app.worker import celery_app

        result = AsyncResult(job_id, app=celery_app)
        return TaskStatusResponse(
            job_id=job_id,
            status=result.status,
            result=result.result if result.successful() else None,
            error=str(result.result) if result.failed() else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete(
    "/documents",
    summary="Delete documents by ID",
    description="Delete specific vectors from Pinecone.",
)
async def delete_documents(body: DeleteRequest):
    try:
        from app.api.deps import get_vector_store

        vs = get_vector_store()
        await vs.delete_documents(body.ids)
        return {"deleted": len(body.ids), "ids": body.ids}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
