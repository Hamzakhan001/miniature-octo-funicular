from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.services.fargate_dispatcher import FargateDispatcher
from app.services.ingestion import IngestionService
from app.services.storage import StorageService


class IngestionEventProcessor:
    def __init__(
        self,
        *,
        storage_service: StorageService,
        ingestion_service: IngestionService,
        job_repository,
        fargate_dispatcher: FargateDispatcher,
    ) -> None:
        self.storage_service = storage_service
        self.ingestion_service = ingestion_service
        self.job_repository = job_repository
        self.fargate_dispatcher = fargate_dispatcher

    async def process(self, payload: dict[str, Any], *, execution_mode: str = "lambda") -> dict[str, Any]:
        job_id = payload["job_id"]

        logger.info(
            "ingestion_event_processor_start",
            job_id=job_id,
            execution_mode=execution_mode,
            processing_target=payload.get("processing_target"),
            object_key=payload.get("object_key"),
        )

        # Create a job record lazily for raw S3 events if it does not already exist.
        existing = self.job_repository.get_job(job_id)
        if existing is None:
            self.job_repository.create_job(
                job_id=job_id,
                status="queued",
                filename=payload["filename"],
                content_type=payload.get("content_type", "application/octet-stream"),
                object_key=payload["object_key"],
                file_size_bytes=payload.get("file_size_bytes", 0),
                processing_target=payload.get("processing_target", "fargate"),
                metadata=payload.get("metadata", {}),
            )
        elif existing.status == "pending_upload":
            self.job_repository.update_status(job_id, status="queued")


        if execution_mode == "fargate":
            return await self._process_file(payload, status="processing_fargate", processing_target="fargate")

        processing_target = payload.get("processing_target", "fargate")
        if processing_target == "fargate":
            self.job_repository.update_status(job_id, status="processing_fargate")
            logger.info("ingestion_dispatching_to_fargate", job_id=job_id)
            dispatch_result = self.fargate_dispatcher.dispatch(payload)
            return {"status": "dispatched", **dispatch_result}

        return await self._process_file(payload, status="processing_lambda", processing_target="lambda")

    async def _process_file(
        self,
        payload: dict[str, Any],
        *,
        status: str,
        processing_target: str,
    ) -> dict[str, Any]:
        job_id = payload["job_id"]
        self.job_repository.update_status(job_id, status=status)

        file_bytes = self.storage_service.read_object_bytes(payload["object_key"])
        ids = await self.ingestion_service.ingest_file(
            file_bytes=file_bytes,
            filename=payload["filename"],
            metadata={
                **payload.get("metadata", {}),
                "job_id": job_id,
                "object_key": payload["object_key"],
                "processing_target": processing_target,
            },
        )

        result = {
            "status": "ok",
            "filename": payload["filename"],
            "chunks": len(ids),
            "ids": ids,
        }
        self.job_repository.update_status(job_id, status="completed", result=result)
        logger.info("ingestion_completed", job_id=job_id, chunks=len(ids), execution_mode=processing_target)
        return result
