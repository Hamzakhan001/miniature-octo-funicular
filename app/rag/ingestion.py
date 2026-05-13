"""Thin wrapper used by ingest.py"""

from app.services.ingestion import IngestionService as IngestionPipeline
from app.core.mdoels import IngestResult
from app.observability.metrics import (INGESTION_CHUNKS_CREATED, INGESTION_BATCHES_TOTAL,
INGESTION_RATE_LIMIT_HITS_TOTAL)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".csv"}

__all__ = ["IngestionPipeline", "SUPPORTED_EXTENSIONS"]