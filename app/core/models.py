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
    text:str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None



class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    guardrail: Optional[GuardrailsResult] = None
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



class DeleteRequest(BaseModel):
    ids: List[str]

