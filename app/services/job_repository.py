from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class IngestionJobRecord:
    job_id: str
    status: str
    filename: str
    content_type: str
    object_key: str
    file_size_bytes: int
    processing_target: str
    upload_url: Optional[str]
    upload_method: Optional[str]
    metadata: dict[str, Any]
    error: Optional[str]
    result: Optional[dict[str, Any]]
    created_at: str
    updated_at: str
    stage_timestamps: dict[str, str]
    progress: dict[str, Any]



class SQLiteJobRepository:
    def __init__(self) -> None:
        settings = get_settings()
        self._db_path = Path(settings.ingestion_queue_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                    content_type TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    processing_target TEXT NOT NULL,
                    upload_url TEXT,
                    upload_method TEXT,
                    metadata_json TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stage_timestamps_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL
                )
                """
            )

            for col,default in [("stage_timestamps_json", "'{}'"),
            ("progress_json", "'{}'")]:
                try:
                    connection.execute(f"ALTER TABLE ingestion_jobs ADD COLUMN {col} {default}")
                except sqlite3.OperationalError:
                    pass

    def create_job(
        self,
        *,
        job_id: str,
        status: str,
        filename: str,
        content_type: str,
        object_key: str,
        file_size_bytes: int,
        processing_target: str,
        upload_url: Optional[str] = None,
        upload_method: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionJobRecord:
        now = utc_now()
        metadata = metadata or {}
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, filename, content_type, object_key,
                    file_size_bytes, processing_target, upload_url, upload_method,
                    metadata_json, error, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    status,
                    filename,
                    content_type,
                    object_key,
                    file_size_bytes,
                    processing_target,
                    upload_url,
                    upload_method,
                    json.dumps(metadata),
                    None,
                    None,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[IngestionJobRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update_status(
        self,
        job_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        upload_url: Optional[str] = None,
        upload_method: Optional[str] = None,
    ) -> IngestionJobRecord:
        existing = self.get_job(job_id)
        if existing is None:
            raise KeyError(f"Unknown ingestion job: {job_id}")

        next_upload_url = upload_url if upload_url is not None else existing.upload_url
        next_upload_method = upload_method if upload_method is not None else existing.upload_method

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?, result_json = ?, upload_url = ?, upload_method = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    error,
                    json.dumps(result) if result is not None else None,
                    next_upload_url,
                    next_upload_method,
                    utc_now(),
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def _row_to_record(self, row: sqlite3.Row) -> IngestionJobRecord:
        return IngestionJobRecord(
            job_id=row["job_id"],
            status=row["status"],
            filename=row["filename"],
            content_type=row["content_type"],
            object_key=row["object_key"],
            file_size_bytes=row["file_size_bytes"],
            processing_target=row["processing_target"],
            upload_url=row["upload_url"],
            upload_method=row["upload_method"],
            metadata=json.loads(row["metadata_json"]),
            error=row["error"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class DynamoDBJobRepository:
    def __init__(self) -> None:
        settings = get_settings()
        self._table_name = settings.job_status_table_name
        if not self._table_name:
            raise ValueError("job_status_table_name is required for DynamoDBJobRepository")

        import boto3

        self._table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
            self._table_name
        )

    def create_job(
        self,
        *,
        job_id: str,
        status: str,
        filename: str,
        content_type: str,
        object_key: str,
        file_size_bytes: int,
        processing_target: str,
        upload_url: Optional[str] = None,
        upload_method: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionJobRecord:
        now = utc_now()
        item = {
            "job_id": job_id,
            "status": status,
            "filename": filename,
            "content_type": content_type,
            "object_key": object_key,
            "file_size_bytes": file_size_bytes,
            "processing_target": processing_target,
            "upload_url": upload_url,
            "upload_method": upload_method,
            "metadata": metadata or {},
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }
        self._table.put_item(Item=item)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[IngestionJobRecord]:
        response = self._table.get_item(Key={"job_id": job_id})
        item = response.get("Item")
        if not item:
            return None
        return IngestionJobRecord(
            job_id=item["job_id"],
            status=item["status"],
            filename=item["filename"],
            content_type=item["content_type"],
            object_key=item["object_key"],
            file_size_bytes=int(item["file_size_bytes"]),
            processing_target=item["processing_target"],
            upload_url=item.get("upload_url"),
            upload_method=item.get("upload_method"),
            metadata=item.get("metadata", {}),
            error=item.get("error"),
            result=item.get("result"),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )

    def update_status(
        self,
        job_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        upload_url: Optional[str] = None,
        upload_method: Optional[str] = None,
    ) -> IngestionJobRecord:
        existing = self.get_job(job_id)
        if existing is None:
            raise KeyError(f"Unknown ingestion job: {job_id}")

        next_upload_url = upload_url if upload_url is not None else existing.upload_url
        next_upload_method = upload_method if upload_method is not None else existing.upload_method

        self._table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #status = :status, #error = :error, #result = :result, "
                "#upload_url = :upload_url, #upload_method = :upload_method, #updated_at = :updated_at"
            ),
            ExpressionAttributeNames={
                "#status": "status",
                "#error": "error",
                "#result": "result",
                "#upload_url": "upload_url",
                "#upload_method": "upload_method",
                "#updated_at": "updated_at",
            },
            ExpressionAttributeValues={
                ":status": status,
                ":error": error,
                ":result": result,
                ":upload_url": next_upload_url,
                ":upload_method": next_upload_method,
                ":updated_at": utc_now(),
            },
        )
        return self.get_job(job_id)
