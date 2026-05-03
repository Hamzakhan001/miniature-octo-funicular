from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.services.fargate_dispatcher import FargateDispatcher
from app.services.ingestion import IngestionService
from app.services.job_repository import SQLiteJobRepository
from app.services.storage import StorageService


class IngestionEventProcessor:
    """Shared processor used by Lambda and the heavier Fargate path."""

    def __init__(
        self,
        *,
        storage_service: StorageService,
        ingestion_service: IngestionService,
        job_repository: SQLiteJobRepository,
        fargate_dispatcher: FargateDispatcher,
    ) -> None:
        self.storage_service = storage_service
        self.ingestion_service = ingestion_service
        self.job_repository = job_repository
        self.fargate_dispatcher = fargate_dispatcher

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = payload["job_id"]
        processing_target = payload["processing_target"]

        if processing_target == "fargate":
            self.job_repository.update_status(job_id, status="processing_fargate")
            dispatch_result = self.fargate_dispatcher.dispatch(payload)
            return {"status": "dispatched", **dispatch_result}

        self.job_repository.update_status(job_id, status="processing_lambda")
        file_bytes = self.storage_service.read_object_bytes(payload["object_key"])
        ids = await self.ingestion_service.ingest_file(
            file_bytes=file_bytes,
            filename=payload["filename"],
            metadata={
                **payload.get("metadata", {}),
                "job_id": job_id,
                "object_key": payload["object_key"],
                "processing_target": "lambda",
            },
        )
        result = {
            "status": "ok",
            "filename": payload["filename"],
            "chunks": len(ids),
            "ids": ids,
        }
        self.job_repository.update_status(job_id, status="completed", result=result)
        logger.info("lambda_ingestion_completed", job_id=job_id, chunks=len(ids))
        return result
