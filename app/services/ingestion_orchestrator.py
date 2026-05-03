from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, UploadFile

from app.core.config import get_settings
from app.core.logging import logger
from app.services.job_repository import IngestionJobRecord, SQLiteJobRepository
from app.services.processing_router import ProcessingRouter
from app.services.queue_backend import QueuePublisher
from app.services.storage import StorageService

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".html", ".json"}


class IngestionOrchestrator:
    """Control-plane service for cloud-style ingestion flows."""

    def __init__(
        self,
        *,
        storage_service: StorageService,
        queue_publisher: QueuePublisher,
        job_repository: SQLiteJobRepository,
        processing_router: ProcessingRouter,
    ) -> None:
        self.settings = get_settings()
        self.storage_service = storage_service
        self.queue_publisher = queue_publisher
        self.job_repository = job_repository
        self.processing_router = processing_router

    def create_upload_session(
        self,
        *,
        filename: str,
        content_type: str,
        file_size_bytes: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionJobRecord:
        self._validate_file(filename=filename, file_size_bytes=file_size_bytes)
        job_id4 = f"job_{uuid.uuid4().hex}"
        target, reason = self.processing_router.pick_target(
            filename=filename,
            file_size_bytes=file_size_bytes,
        )
        upload_target = self.storage_service.generate_upload_target(
            filename=filename,
            content_type=content_type or "application/octet-stream",
            job_id=job_id,
        )
        job = self.job_repository.create_job(
            job_id=job_id,
            status="pending_upload",
            filename=filename,
            content_type=content_type or "application/octet-stream",
            object_key=upload_target.object_key,
            file_size_bytes=file_size_bytes,
            processing_target=target,
            upload_url=upload_target.upload_url,
            upload_method=upload_target.upload_method,
            metadata={**(metadata or {}), "routing_reason": reason},
        )
        logger.info(
            "upload_session_created",
            job_id=job.job_id,
            filename=job.filename,
            processing_target=job.processing_target,
            object_key=job.object_key,
        )
        return job

    async def upload_and_enqueue(
        self,
        *,
        upload_file: UploadFile,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionJobRecord:
        filename = upload_file.filename or "upload"
        declared_content_type = upload_file.content_type or "application/octet-stream"
        self._validate_file(filename=filename, file_size_bytes=None)
        job_id = f"job_{uuid.uuid4().hex}"
        object_key = self.storage_service.build_object_key(filename, job_id)

        stored = await self.storage_service.store_upload(
            upload_file=upload_file,
            object_key=object_key,
        )
        self._validate_file(filename=filename, file_size_bytes=stored.file_size_bytes)

        target, reason = self.processing_router.pick_target(
            filename=filename,
            file_size_bytes=stored.file_size_bytes,
        )
        job = self.job_repository.create_job(
            job_id=job_id,
            status="uploaded",
            filename=filename,
            content_type=declared_content_type,
            object_key=stored.object_key,
            file_size_bytes=stored.file_size_bytes,
            processing_target=target,
            metadata={
                **(metadata or {}),
                "checksum_sha256": stored.checksum_sha256,
                "routing_reason": reason,
            },
        )
        self._publish_ingestion_event(job)
        return self.job_repository.update_status(job.job_id, status="queued")

    def get_status(self, job_id: str) -> IngestionJobRecord:
        job = self.job_repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found")
        return job

    def _publish_ingestion_event(self, job: IngestionJobRecord) -> None:
        payload = {
            "job_id": job.job_id,
            "filename": job.filename,
            "content_type": job.content_type,
            "object_key": job.object_key,
            "file_size_bytes": job.file_size_bytes,
            "processing_target": job.processing_target,
            "metadata": job.metadata,
        }
        publish_result = self.queue_publisher.publish(payload)
        logger.info(
            "ingestion_event_emitted",
            job_id=job.job_id,
            processing_target=job.processing_target,
            queue_backend=publish_result.get("backend"),
        )

    def _validate_file(self, *, filename: str, file_size_bytes: Optional[int]) -> None:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        if file_size_bytes is not None:
            max_bytes = self.settings.ingestion_max_file_size_mb * 1024 * 1024
            if file_size_bytes <= 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if file_size_bytes > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {self.settings.ingestion_max_file_size_mb}MB",
                )
