from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import logger
from app.services.ingestion import IngestionService


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso8601(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def from_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    status: str
    filename: str
    staged_path: str
    file_size_bytes: int
    checksum_sha256: str
    attempts: int
    max_attempts: int
    metadata: dict[str, Any]
    result: Optional[dict[str, Any]]
    error: Optional[str]
    available_at: str
    created_at: str
    updated_at: str


class LocalIngestionQueue:
    """Durable local queue that mirrors the control flow we'd later move to SQS."""

    def __init__(self, ingestion_service: IngestionService) -> None:
        self.settings = get_settings()
        self.ingestion_service = ingestion_service
        self._db_path = Path(self.settings.ingestion_queue_db_path)
        self._staging_dir = Path(self.settings.ingestion_staging_dir)
        self._processed_dir = Path(self.settings.ingestion_processed_dir)
        self._poll_interval = self.settings.ingestion_queue_poll_interval_seconds
        self._worker_concurrency = max(1, self.settings.ingestion_worker_concurrency)
        self._stop_event = asyncio.Event()
        self._worker_tasks: list[asyncio.Task[None]] = []

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    async def start(self) -> None:
        if self._worker_tasks:
            return

        self._stop_event.clear()
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(worker_id))
            for worker_id in range(self._worker_concurrency)
        ]
        logger.info("ingestion_queue_started", workers=self._worker_concurrency)

    async def stop(self) -> None:
        if not self._worker_tasks:
            return

        self._stop_event.set()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        logger.info("ingestion_queue_stopped")

    async def enqueue(
        self,
        *,
        filename: str,
        staged_path: str,
        file_size_bytes: int,
        checksum_sha256: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionJob:
        metadata = metadata or {}
        now = utc_now()
        job = IngestionJob(
            job_id=f"job_{uuid.uuid4().hex}",
            status="queued",
            filename=filename,
            staged_path=staged_path,
            file_size_bytes=file_size_bytes,
            checksum_sha256=checksum_sha256,
            attempts=0,
            max_attempts=self.settings.ingestion_queue_max_attempts,
            metadata=metadata,
            result=None,
            error=None,
            available_at=to_iso8601(now),
            created_at=to_iso8601(now),
            updated_at=to_iso8601(now),
        )
        await asyncio.to_thread(self._insert_job_sync, job)
        logger.info(
            "ingestion_job_enqueued",
            job_id=job.job_id,
            filename=filename,
            size_bytes=file_size_bytes,
        )
        return job

    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return await asyncio.to_thread(self._get_job_sync, job_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    staged_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _insert_job_sync(self, job: IngestionJob) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, filename, staged_path, file_size_bytes,
                    checksum_sha256, attempts, max_attempts, metadata_json, result_json,
                    error, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status,
                    job.filename,
                    job.staged_path,
                    job.file_size_bytes,
                    job.checksum_sha256,
                    job.attempts,
                    job.max_attempts,
                    json.dumps(job.metadata),
                    json.dumps(job.result) if job.result is not None else None,
                    job.error,
                    job.available_at,
                    job.created_at,
                    job.updated_at,
                ),
            )

    def _get_job_sync(self, job_id: str) -> Optional[IngestionJob]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def _claim_next_job_sync(self) -> Optional[IngestionJob]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE status = 'queued' AND available_at <= ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (to_iso8601(utc_now()),),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None

            job = self._row_to_job(row)
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, attempts = ?, updated_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (
                    "processing",
                    job.attempts + 1,
                    to_iso8601(utc_now()),
                    job.job_id,
                ),
            )
            connection.execute("COMMIT")
            job.status = "processing"
            job.attempts += 1
            job.updated_at = to_iso8601(utc_now())
            job.error = None
            return job

    def _mark_completed_sync(self, job_id: str, result: dict[str, Any]) -> None:
        now = to_iso8601(utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, result_json = ?, error = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                ("completed", json.dumps(result), now, job_id),
            )

    def _mark_retryable_sync(self, job: IngestionJob, error: str, delay_seconds: float) -> None:
        now = utc_now()
        available_at = to_iso8601(now + timedelta(seconds=delay_seconds))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?, available_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("queued", error, available_at, to_iso8601(now), job.job_id),
            )

    def _mark_failed_sync(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("failed", error, to_iso8601(utc_now()), job_id),
            )

    def _row_to_job(self, row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            job_id=row["job_id"],
            status=row["status"],
            filename=row["filename"],
            staged_path=row["staged_path"],
            file_size_bytes=row["file_size_bytes"],
            checksum_sha256=row["checksum_sha256"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            metadata=json.loads(row["metadata_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            available_at=row["available_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def _worker_loop(self, worker_id: int) -> None:
        while not self._stop_event.is_set():
            job = await asyncio.to_thread(self._claim_next_job_sync)
            if job is None:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    continue
                continue

            await self._process_job(worker_id, job)

    async def _process_job(self, worker_id: int, job: IngestionJob) -> None:
        logger.info(
            "ingestion_job_processing",
            worker_id=worker_id,
            job_id=job.job_id,
            filename=job.filename,
            attempt=job.attempts,
        )
        try:
            ids = await self.ingestion_service.ingest_staged_file(
                staged_path=job.staged_path,
                filename=job.filename,
                metadata={
                    **job.metadata,
                    "job_id": job.job_id,
                    "checksum_sha256": job.checksum_sha256,
                },
            )
            result = {
                "status": "ok",
                "filename": job.filename,
                "chunks": len(ids),
                "ids": ids,
            }
            await asyncio.to_thread(self._mark_completed_sync, job.job_id, result)
            logger.info(
                "ingestion_job_completed",
                worker_id=worker_id,
                job_id=job.job_id,
                filename=job.filename,
                chunks=len(ids),
            )
        except Exception as exc:
            error_message = str(exc)
            if job.attempts >= job.max_attempts:
                await asyncio.to_thread(self._mark_failed_sync, job.job_id, error_message)
                logger.error(
                    "ingestion_job_failed",
                    worker_id=worker_id,
                    job_id=job.job_id,
                    filename=job.filename,
                    attempts=job.attempts,
                    error=error_message,
                )
                return

            delay_seconds = self._compute_retry_delay_seconds(job.attempts)
            await asyncio.to_thread(self._mark_retryable_sync, job, error_message, delay_seconds)
            logger.warning(
                "ingestion_job_requeued",
                worker_id=worker_id,
                job_id=job.job_id,
                filename=job.filename,
                attempts=job.attempts,
                retry_in_seconds=round(delay_seconds, 2),
                error=error_message,
            )

    def _compute_retry_delay_seconds(self, attempts: int) -> float:
        base = max(1, self.settings.ingestion_base_retry_seconds)
        exponential_delay = base * (2 ** max(0, attempts - 1))
        jitter = random.uniform(0, base)
        return float(exponential_delay + jitter)
