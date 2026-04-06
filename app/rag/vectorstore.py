"""Thin wrapper used by ingest.py/query.py routes (app.rag.vector_store.VectorS)"""
from app.services.vector_store import VectorStoreService as VectorStore


__all__ = ["VectorStore"]