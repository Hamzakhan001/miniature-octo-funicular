from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.core.logging import logger


def run_payload_in_background(
    *,
    payload: dict[str, Any],
    execution_mode: str,
) -> None:
    thread = threading.Thread(
        target=_run_payload,
        kwargs={"payload": payload, "execution_mode": execution_mode},
        daemon=True,
        name=f"ingestion-{execution_mode}-{payload.get('job_id', 'unknown')}",
    )
    thread.start()


def _run_payload(*, payload: dict[str, Any], execution_mode: str) -> None:
    from app.api.deps import get_ingestion_event_processor, get_job_repository

    processor = get_ingestion_event_processor()
    job_repository = get_job_repository()
    job_id = payload.get("job_id", "unknown")

    try:
        asyncio.run(processor.process(payload, execution_mode=execution_mode))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception(
            "local_ingestion_background_failed",
            job_id=job_id,
            execution_mode=execution_mode,
            error=str(exc),
        )
        try:
            job_repository.update_status(job_id, status="failed", error=str(exc))
        except Exception:
            logger.exception(
                "local_ingestion_failure_status_update_failed",
                job_id=job_id,
            )
