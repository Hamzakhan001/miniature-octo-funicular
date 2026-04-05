from __future__ import annotations
from functools import lru_cache
from typing import List, Literal, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    open_ai_api_key: str = Field(..., description="OpenAI API key")
    openai_chat_model: str = "gpt-4o"
    chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-ada-002"
    embedding_model:str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    chat_temperature: float = 0.0
    max_tokens: int = 1024


    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_region: str = "us-east-1"
    pinecone_environment: str = "us-east-1"
    pinecone_index: str = "rag-prod"
    pinecone_index_name: str = "rag-prod"


    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    top_k: int = 8

    redis_url: str = "redis: //localhost:6379/0"
    cache_ttl_seconds: int = 3600

    app_env: Literal["development", "production", "test"] = "development"
    cors_origins: Any = "http//localhost:3000"
    api_key_header: str = "X-API-Key"
    api_keys: Any = ""
    app_secret_key: str = "change-me"
    log_level: str = "INFO"

    input_max_chars: int = 4000
    blocked_topics: Any = "violence, self-harm, illegal weapons"
    pii_detection: bool = True
    output_max_chars: int = 8000
    hallucination_threshold: float = 0.25

    otlp_endpoint: str = "http://localhost: 4317"
    enable_tracing: bool = True

    eval_faithfulness_threshold: float = 0.7
    eval_relevance_threahold: float = 0.7
    golden_set_path: str = "data/golden_set.json"


    @field_validator("cors_origins", "blocked_topics", "api_keys", mode="before")
    @classmethod
    def parse_str_to_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
        

@lru_cache
def get_settings() -> Settings:
    return Settings()