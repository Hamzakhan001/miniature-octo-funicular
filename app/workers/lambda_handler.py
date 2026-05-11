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
        logger.info(
            "lambda_enriched_payload_received",
            job_id=body.get("job_id"),
            object_key=body.get("object_key"),
        )
        return [body]

    # Raw S3 -> SQS event
    payloads: list[dict[str, Any]] = []
    for s3_record in body.get("Records", []):
        s3 = s3_record.get("s3", {})
        bucket = s3.get("bucket", {})
        obj = s3.get("object", {})
        key = obj.get("key")
        if not key:
            logger.warning("lambda_s3_record_missing_key")
            continue

        parts = key.split("/")
        if len(parts) < 3:
            logger.warning("invalid_s3_object_key", key=key)
            continue

        job_id = parts[1]
        filename = parts[-1]
        size = obj.get("size", 0)

        logger.info(
            "lambda_s3_payload_built",
            job_id=job_id,
            key=key,
            bucket=bucket.get("name"),
            filename=filename,
            file_size_bytes=size,
        )

        payloads.append(
            {
                "job_id": job_id,
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

    logger.info(
        "lambda_event_received",
        record_count=len(event.get("Records", [])),
    )

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            payloads = _extract_payloads(record)
            logger.info(
                "lambda_payloads_extracted",
                message_id=message_id,
                payload_count=len(payloads),
            )

            for payload in payloads:
                logger.info(
                    "lambda_processing_payload",
                    message_id=message_id,
                    job_id=payload.get("job_id"),
                    object_key=payload.get("object_key"),
                    processing_target=payload.get("processing_target"),
                )
                asyncio.run(processor.process(payload))
                processed += 1
                logger.info(
                    "lambda_payload_processed",
                    message_id=message_id,
                    job_id=payload.get("job_id"),
                )
        except Exception as exc:
            logger.exception(
                "lambda_ingestion_failed",
                message_id=message_id,
                error=str(exc),
            )
            failures.append({"itemIdentifier": message_id})

    logger.info(
        "lambda_event_completed",
        processed=processed,
        failure_count=len(failures),
    )
    return {"batchItemFailures": failures, "processed": processed}
