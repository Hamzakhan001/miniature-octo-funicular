""""
Ingestion routes sync text ingestion + async file ingestion via celery

Routes:
    POST /api/v1/ingest/text
    POST /api/v1/ingest/file        via celery queue -> returns job id
    GET /api/v1/ingest/status/{id}   poll celery task status
    DELETE /api/v1/ingest/documents  delete vectors by ID

Design Decisions:
Text Ingestion: fast enough (<5s) to await directly in route hanlder
File Ingestion: can take 30-120seconds for large pdfs, so offloaded to celery to not block the request
                Returns job ID immediately for polling


Why Not FastAPI Background Tasks?
    - No way to track progress or status
    - No way to cancel or retry
    - No way to handle failures gracefully
    - No way to queue multiple tasks

"""


from __future__ import annotations
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Any,Dict, List, Optional

router = APIRouter(prefix="/ingest", tags=["ingestion"])

MAX_FILE_SIZE = 50*1024*1024

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".html"}

SYNC_THREASHOLD_BYTES = 1*1024*1024

class AsyncIngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message : str
    check_status_url: str


class TaskStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    


def get_ingestion():
    from app.api.deps import get_ingestion_service
    return get_ingestion_service()


@router.post("/text", response_model = IngestResult, summary="Ingest raw text", description="Ingest raw text directly into the vector store. Fast awaited synchronous operation.")
async def ingest_text(body: IngestTextRequest):
    try:
        ingestion = get_ingestion()
        ids = await ingestion.ingest_text(body.text, body.metadata, source=body.source)
        return IngestResult(status="ok", chunks=len(ids), ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

 
@router.post("/file", summary="Ingest a file", description="Upload and ingest a file asynchronously via celery. Returns job ID for polling status.")
async def ingest_file(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {SUPPORTED_EXTENSIONS}")

        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB")
        
        filename = file.filename or "upload"

        if len(content) < SYNC_THREASHOLD_BYTES:
            try:
                ingestion = get_ingestion()
                ids = await ingestion.ingest_file(file.file, filename)
                return IngestResult(status="ok", chunks=len(ids), ids=ids)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        job_id = f"job_{int(time.time())}"

        try:
            from app.worker import ingest_file_task

            task = ingest_file_task.delay(content.hex(), filename, {"source": filename, "queued_At": time.time()})
            return AsyncIngestResponse(
                job_id=job_id,
                status="queued",
                filename=filename,
                message="File queued for processing.",
                check_status_url=f"/api/v1/ingest/status/{job_id}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        return AsyncIngestResponse(
            job_id=job_id,
            status="accepted",
            filename=file.filename or "unknown",
            message="File received. Processing will start shortly.",
            check_status_url=f"/api/v1/ingest/status/{job_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model= TaskStatusResponse, summary="check ingestion job status", descripiton = "Poll this endpoint")
def get_task_status(job_id: str):
    try:
        from app.worker import celery_app
        from celery.result import AsyncResult
        
        result = AsyncResult(job_id, app=celery_app)
        
        return TaskStatusResponse(
            job_id=job_id,
            status=result.status,
            result=result.result if result.successful() else None,
            error = str(result.result) if result.failed() else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents", summary="Delete documents by id", description="delete specific vectors from pinecone")
def delete_docuements(body: DeleteRequest):
    try:
        from app.api.deps import get_vector_store
        vs = get_vector_store()
        await vs.delete_docuements(body.ids)
        return {"deleted": len(body.ids), "ids": body.ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))