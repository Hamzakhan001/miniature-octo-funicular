from __future__ import annotations

import asyncio
import json
import os

from app.api.deps import get_ingestion_event_processor
from app.core.logging import logger, setup_logging


def main() -> None:
    setup_logging()
    raw_payload = os.environ.get("INGESTION_EVENT")
    if not raw_payload:
        raise RuntimeError("INGESTION_EVENT is required for the Fargate worker")

    payload = json.loads(raw_payload)
    processor = get_ingestion_event_processor()
    asyncio.run(processor.process({**payload, "processing_target": "lambda"}))
    logger.info("fargate_ingestion_completed", job_id=payload.get("job_id"))


if __name__ == "__main__":
    main()
