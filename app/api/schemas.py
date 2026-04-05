from pydantic import BaseModel
from typing import Optional, List, Any

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    filter: Optional[dict] = None
    stram: bool = False
    use_rerank: bool= False


class SourceDocument(BaseMoel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    source: str = "manual"
    metadata: Optional[dict] = None

class IngestResponse(BaseModel):
    ids: List[str]
    count: int
    message: str


class HealthResponse(BaseModel):
    status: str
    env: str
    pinecone_index: str
    openai_model: str

class DeleteRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1)

class DeleteResponse(BaseModel):
    deleted: int
    message: str
