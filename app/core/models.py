from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    env: str
    index: str
    model: str


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"


class GuardrailResult(BaseModel):
    action: GuardrailAction
    reason: Optional[str] = None
    redacted_text: Optional[str] = None
    latency_ms: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    filter: Optional[Dict[str, Any]] = None
    run_eval: bool = False


class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    guardrail: Optional[GuardrailResult] = None
    eval_scores: Optional[Dict[str, float]] = None
    cached: bool = False
    latency_ms: float = 0.0


class IngestionTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    source: str = "manual"
    metadata: Optional[Dict[str, Any]] = None


class IngestionResult(BaseModel):
    status: str = "ok"
    chunks: int
    ids: List[str]


class ProcessingTarget(str, Enum):
    LAMBDA = "lambda"
    FARGATE = "fargate"


class UploadInitRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(default="application/octet-stream", max_length=255)
    file_size_bytes: int = Field(..., gt=0, le=100 * 1024 * 1024)
    metadata: Optional[Dict[str, Any]] = None


class UploadInitResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    object_key: str
    processing_target: ProcessingTarget
    upload_url: Optional[str] = None
    upload_method: Optional[str] = None
    expires_in_seconds: Optional[int] = None


class IngestionJobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    object_key: str
    processing_target: ProcessingTarget
    file_size_bytes: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class DeleteRequest(BaseModel):
    ids: List[str]
