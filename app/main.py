from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import AuthMiddleware, RequestIDMiddleware, RequestLoggingMiddleware
from app.api.routes import audit, evaluation, health, ingest, query
from app.core.logging import setup_logging
from app.observability import configure_metrics, configure_tracing
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    configure_tracing()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    cors_origins = settings.cors_origins or ["*"]
    allow_credentials = "*" not in cors_origins

    app = FastAPI(
        title="Production RAG API",
        description="Retrieval-Augmented Generation with guardrails, evaluation and observability.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuthMiddleware)

    prefix = "/api/v1"
    app.include_router(query.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(evaluation.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)
    app.include_router(audit.router, prefix=prefix)
    configure_metrics(app)
    return app


app = create_app()
