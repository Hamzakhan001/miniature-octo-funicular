from __future__ import annotations

import asyncio
import json
from typing import Any

from app.api.deps import get_ingestion_event_processor
from app.core.logging import logger, setup_logging


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint for SQS-triggered ingestion events."""
    setup_logging()
    processor = get_ingestion_event_processor()
    processed = 0
    failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            payload = json.loads(record["body"])
            asyncio.run(processor.process(payload))
            processed += 1
        except Exception as exc:
            logger.exception("lambda_ingestion_failed", message_id=message_id, error=str(exc))
            failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": failures, "processed": processed}
