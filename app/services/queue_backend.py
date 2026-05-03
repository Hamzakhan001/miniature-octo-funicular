from __future__ import annotations

import json
from typing import Any

from app.core.config import get_settings
from app.core.logging import logger


class QueuePublisher:
    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class MemoryQueuePublisher(QueuePublisher):
    """Development fallback that records the event in logs without pretending to scale."""

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info("ingestion_event_recorded_locally", payload=payload)
        return {"backend": "memory", "accepted": True}


class SQSQueuePublisher(QueuePublisher):
    def __init__(self) -> None:
        settings = get_settings()
        self._queue_url = settings.sqs_ingestion_queue_url
        self._region = settings.aws_region

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for the SQS queue backend") from exc

        self._client = boto3.client("sqs", region_name=self._region)

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps(payload),
        )
        logger.info(
            "ingestion_event_published",
            queue_backend="sqs",
            message_id=response.get("MessageId"),
            job_id=payload.get("job_id"),
            processing_target=payload.get("processing_target"),
        )
        return {"backend": "sqs", "message_id": response.get("MessageId"), "accepted": True}



