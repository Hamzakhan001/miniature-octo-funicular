""""Thin wrapper used by ingest.py"""

from app.services.ingestion import IngestionService as IngestionPipeline
from app.core.mdoels import IngestResult

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".csv"}

__all__ = ["IngestionPipeline", "SUPPORTED_EXTENSIONS"]