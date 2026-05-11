from __future__ import annotations
from functools import lru_cache
from typing import List, Literal, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_chat_model: str = "gpt-4o"
    chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-ada-002"
    embedding_model:str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    chat_temperature: float = 0.0
    max_tokens: int = 1024
    job_status_backend: Literal["sqlite", "dynamodb"] = "sqlite"
    job_status_table_name: str = ""
    openai_api_key_secret_arn: str = ""
    vector_store_api_key_secret_arn: str = ""



    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_region: str = "us-east-1"
    pinecone_environment: str = "us-east-1"
    pinecone_index: str = "rag-prod"
    pinecone_index_name: str = "rag-prod"


    chunk_size: int = 384
    chunk_overlap: int = 64
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    top_k: int = 8

    redis_url: str = "redis: //localhost:6379/0"
    cache_ttl_seconds: int = 3600

    storage_backend: Literal["local", "s3"] = "local"
    queue_backend: Literal["memory", "sqs"] = "memory"
    auto_process_local_ingestion: bool = False
    aws_region: str = "eu-west-2"
    s3_ingestion_bucket: str = ""
    s3_presign_expiration_seconds: int = 900
    sqs_ingestion_queue_url: str = ""
    ecs_cluster: str = ""
    ecs_task_definition: str = ""
    ecs_container_name: str = ""
    ecs_subnets: Any = ""
    ecs_security_groups: Any = ""
    ecs_assign_public_ip: bool = False
    lambda_max_inline_file_size_mb: int = 12
    lambda_supported_extensions: Any = ".txt,.md,.html,.json"
    fargate_preferred_extensions: Any = ".pdf,.docx,.csv"

    ingestion_max_file_size_mb: int = 100
    ingestion_sync_threshold_bytes: int = 1048576
    ingestion_queue_db_path: str = "data/ingestion_queue/jobs.db"
    ingestion_staging_dir: str = "data/ingestion_queue/uploads"
    ingestion_processed_dir: str = "data/ingestion_queue/processed"
    ingestion_queue_poll_interval_seconds: float = 1.0
    ingestion_queue_max_attempts: int = 5
    ingestion_worker_concurrency: int = 2
    ingestion_base_retry_seconds: int = 5

    app_env: Literal["development", "production", "test"] = "development"
    cors_origins: Any = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )
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


    @field_validator(
        "cors_origins",
        "blocked_topics",
        "api_keys",
        "ecs_subnets",
        "ecs_security_groups",
        "lambda_supported_extensions",
        "fargate_preferred_extensions",
        mode="before",
    )
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
