from functools import lru_cache
from app.core.config import get_settings
from app.services.event_processor import IngestionEventProcessor
from app.services.fargate_dispatcher import ECSFargateDispatcher, NoopFargateDispatcher
from app.services.ingestion_orchestrator import IngestionOrchestrator
from app.services.vector_store import VectorStoreService
from app.services.ingestion import IngestionService
from app.services.job_repository import SQLiteJobRepository, DynamoDBJobRepository
from app.services.processing_router import ProcessingRouter
from app.services.queue_backend import MemoryQueuePublisher, SQSQueuePublisher
from app.services.rag_chain import RAGService
from app.services.storage import LocalStorageService, S3StorageService


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()

@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(vector_store = get_vector_store())

@lru_cache
def get_job_repository():
    settings = get_settings()
    if settings.job_status_backend == "dynamodb":
        return DynamoDBJobRepository()
    return SQLiteJobRepository()


@lru_cache
def get_processing_router() -> ProcessingRouter:
    return ProcessingRouter()

@lru_cache
def get_storage_service():
    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3StorageService()
    return LocalStorageService()

@lru_cache
def get_queue_publisher():
    settings = get_settings()
    if settings.queue_backend == "sqs":
        return SQSQueuePublisher()
    return MemoryQueuePublisher()

@lru_cache
def get_fargate_dispatcher():
    settings = get_settings()
    if settings.ecs_cluster and settings.ecs_task_definition and settings.ecs_container_name:
        return ECSFargateDispatcher()
    return NoopFargateDispatcher()

@lru_cache
def get_ingestion_orchestrator() -> IngestionOrchestrator:
    return IngestionOrchestrator(
        storage_service=get_storage_service(),
        queue_publisher=get_queue_publisher(),
        job_repository=get_job_repository(),
        processing_router=get_processing_router(),
    )

@lru_cache
def get_ingestion_event_processor() -> IngestionEventProcessor:
    return IngestionEventProcessor(
        storage_service=get_storage_service(),
        ingestion_service=get_ingestion_service(),
        job_repository=get_job_repository(),
        fargate_dispatcher=get_fargate_dispatcher(),
    )

@lru_cache
def get_rag_chain_service() -> RAGService:
    return RAGService(vector_store = get_vector_store())
