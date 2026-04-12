from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


from pydantic import BaseModel, Field

class QueryAuditRecord(BaseModel):
    event_type: str = "query"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    question: str
    top_k: Optional[int] = None
    retrieval_method: str 
    docs_retrieved: int = 0
    sources: list[str] = Field(default_factory=list)
    answer_length: int = 0
    outcome: str
    run_eval: bool = False
    eval_scores: Optional[dict[str, float]] = None
    total_latency_ms: float = 0.0
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)



class IngestionAuditRecord(BaseModel):
    event_type: str = "ingestion"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_name: str
    source_type: str
    status: str
    chunks_created: int = 0
    vectors_upserted: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)