from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.models import (
    DeleteRequest,
    IngestionJobStatusResponse,
    IngestionResult,
    IngestionTextRequest,
    ProcessingTarget,
    UploadInitRequest,
    UploadInitResponse,
)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


def get_ingestion():
    from app.api.deps import get_ingestion_service

    return get_ingestion_service()


def get_orchestrator():
    from app.api.deps import get_ingestion_orchestrator

    return get_ingestion_orchestrator()


@router.post(
    "/text",
    response_model=IngestionResult,
    summary="Ingest raw text",
    description="Ingest raw text directly into the vector store.",
)
async def ingest_text(body: IngestionTextRequest):
    try:
        ingestion = get_ingestion()
        ids = await ingestion.ingest_text(body.text, source=body.source, metadata=body.metadata)
        return IngestionResult(status="ok", chunks=len(ids), ids=ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/uploads/create",
    response_model=UploadInitResponse,
    summary="Create a direct upload session",
    description="Create an event-driven ingestion job and, when S3 is configured, return a presigned upload URL.",
)
async def create_upload(body: UploadInitRequest):
    try:
        orchestrator = get_orchestrator()
        job = orchestrator.create_upload_session(
            filename=body.filename,
            content_type=body.content_type,
            file_size_bytes=body.file_size_bytes,
            metadata=body.metadata,
        )
        settings = get_settings()
        return UploadInitResponse(
            job_id=job.job_id,
            status=job.status,
            filename=job.filename,
            object_key=job.object_key,
            processing_target=ProcessingTarget(job.processing_target),
            upload_url=job.upload_url,
            upload_method=job.upload_method,
            expires_in_seconds=settings.s3_presign_expiration_seconds if job.upload_url else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/file",
    response_model=IngestionJobStatusResponse,
    summary="Upload a file through the API and enqueue it for asynchronous processing",
    description="Compatibility route for server-mediated uploads. Production frontends should prefer direct-to-S3 upload sessions.",
)
async def ingest_file(file: UploadFile = File(...)):
    try:
        orchestrator = get_orchestrator()
        job = await orchestrator.upload_and_enqueue(upload_file=file)
        return IngestionJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            filename=job.filename,
            object_key=job.object_key,
            processing_target=ProcessingTarget(job.processing_target),
            file_size_bytes=job.file_size_bytes,
            result=job.result,
            error=job.error,
            metadata=job.metadata,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/status/{job_id}",
    response_model=IngestionJobStatusResponse,
    summary="Check ingestion job status",
    description="Return the tracked control-plane status for an ingestion job.",
)
def get_task_status(job_id: str):
    try:
        orchestrator = get_orchestrator()
        job = orchestrator.get_status(job_id)
        return IngestionJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            filename=job.filename,
            object_key=job.object_key,
            processing_target=ProcessingTarget(job.processing_target),
            file_size_bytes=job.file_size_bytes,
            result=job.result,
            error=job.error,
            metadata=job.metadata,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
    except HTTPException:
        raise
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
