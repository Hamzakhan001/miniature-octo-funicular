from functools import lru_cache
from app.services.vector_store import VectorStoreService
from app.services.ingestion import IngestionService
from app.services.rag_chain import RAGService


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()

@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(vector_store = get_vector_store())

@lru_cache
def get_rag_chain_service() -> RAGService:
    return RAGService(vector_store = get_vector_store())

