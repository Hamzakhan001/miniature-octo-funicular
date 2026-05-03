from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from app.core.config import get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class StoredObject:
    def __init__(
        self,
        *,
        object_key: str,
        content_type: str,
        file_size_bytes: int,
        checksum_sha256: str,
        upload_url: Optional[str] = None,
        upload_method: Optional[str] = None,
    ) -> None:
        self.object_key = object_key
        self.content_type = content_type
        self.file_size_bytes = file_size_bytes
        self.checksum_sha256 = checksum_sha256
        self.upload_url = upload_url
        self.upload_method = upload_method


class StorageService:
    def build_object_key(self, filename: str, job_id: str) -> str:
        raise NotImplementedError

    def generate_upload_target(
        self,
        *,
        filename: str,
        content_type: str,
        job_id: str,
    ) -> StoredObject:
        raise NotImplementedError

    async def store_upload(
        self,
        *,
        upload_file: UploadFile,
        object_key: str,
    ) -> StoredObject:
        raise NotImplementedError

    def read_object_bytes(self, object_key: str) -> bytes:
        raise NotImplementedError


def sanitize_filename(filename: str) -> str:
    base = Path(filename or "upload").name
    sanitized = _SAFE_FILENAME_RE.sub("_", base).strip("._")
    return sanitized or f"upload_{uuid.uuid4().hex}"


class LocalStorageService(StorageService):
    def __init__(self) -> None:
        settings = get_settings()
        self._root = Path(settings.ingestion_staging_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def build_object_key(self, filename: str, job_id: str) -> str:
        return f"raw/local/{job_id}/{sanitize_filename(filename)}"

    def generate_upload_target(
        self,
        *,
        filename: str,
        content_type: str,
        job_id: str,
    ) -> StoredObject:
        object_key = self.build_object_key(filename, job_id)
        return StoredObject(
            object_key=object_key,
            content_type=content_type,
            file_size_bytes=0,
            checksum_sha256="",
            upload_url=None,
            upload_method=None,
        )

    async def store_upload(
        self,
        *,
        upload_file: UploadFile,
        object_key: str,
    ) -> StoredObject:
        destination = self._root / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)

        hasher = hashlib.sha256()
        total_bytes = 0
        content_type = upload_file.content_type or "application/octet-stream"
        with destination.open("wb") as target:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)
                target.write(chunk)

        await upload_file.close()
        return StoredObject(
            object_key=object_key,
            content_type=content_type,
            file_size_bytes=total_bytes,
            checksum_sha256=hasher.hexdigest(),
        )

    def read_object_bytes(self, object_key: str) -> bytes:
        source = self._root / object_key
        return source.read_bytes()


class S3StorageService(StorageService):
    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.s3_ingestion_bucket
        self._region = settings.aws_region
        self._presign_expiration_seconds = settings.s3_presign_expiration_seconds

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for the S3 storage backend") from exc

        self._client = boto3.client("s3", region_name=self._region)

    def build_object_key(self, filename: str, job_id: str) -> str:
        return f"raw/{job_id}/{sanitize_filename(filename)}"

    def generate_upload_target(
        self,
        *,
        filename: str,
        content_type: str,
        job_id: str,
    ) -> StoredObject:
        object_key = self.build_object_key(filename, job_id)
        upload_url = self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": object_key,
                "ContentType": content_type,
            },
            ExpiresIn=self._presign_expiration_seconds,
        )
        return StoredObject(
            object_key=object_key,
            content_type=content_type,
            file_size_bytes=0,
            checksum_sha256="",
            upload_url=upload_url,
            upload_method="PUT",
        )

    async def store_upload(
        self,
        *,
        upload_file: UploadFile,
        object_key: str,
    ) -> StoredObject:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for the S3 storage backend") from exc

        temp_path = Path("/tmp") / f"{uuid.uuid4().hex}_{sanitize_filename(upload_file.filename or 'upload')}"
        hasher = hashlib.sha256()
        total_bytes = 0
        with temp_path.open("wb") as target:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)
                target.write(chunk)

        await upload_file.close()
        content_type = upload_file.content_type or "application/octet-stream"
        with temp_path.open("rb") as source:
            self._client.upload_fileobj(
                Fileobj=source,
                Bucket=self._bucket,
                Key=object_key,
                ExtraArgs={"ContentType": content_type},
            )
        os.unlink(temp_path)

        return StoredObject(
            object_key=object_key,
            content_type=content_type,
            file_size_bytes=total_bytes,
            checksum_sha256=hasher.hexdigest(),
        )

    def read_object_bytes(self, object_key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=object_key)
        return response["Body"].read()
