from __future__ import annotations

import asyncio
import json
from typing import Any

from app.api.deps import get_ingestion_event_processor
from app.core.logging import logger, setup_logging


def _extract_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    body = json.loads(record["body"])

    # Enriched app-produced message
    if "object_key" in body:
        return [body]

    # Raw S3 -> SQS event
    payloads: list[dict[str, Any]] = []
    for s3_record in body.get("Records", []):
        s3 = s3_record.get("s3", {})
        bucket = s3.get("bucket", {})
        obj = s3.get("object", {})
        key = obj.get("key")
        if not key:
            continue

        filename = key.split("/")[-1]
        size = obj.get("size", 0)

        payloads.append(
            {
                "job_id": f"s3::{bucket.get('name','unknown')}::{key}",
                "filename": filename,
                "object_key": key,
                "file_size_bytes": size,
                "processing_target": "fargate",
                "metadata": {
                    "bucket": bucket.get("name"),
                    "event_name": s3_record.get("eventName"),
                    "event_time": s3_record.get("eventTime"),
                    "source": "s3_event",
                },
            }
        )
    return payloads


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    setup_logging()
    processor = get_ingestion_event_processor()
    processed = 0
    failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            payloads = _extract_payloads(record)
            for payload in payloads:
                asyncio.run(processor.process(payload))
                processed += 1
        except Exception as exc:
            logger.exception("lambda_ingestion_failed", message_id=message_id, error=str(exc))
            failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": failures, "processed": processed}
