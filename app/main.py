from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import query, ingest, evaluation, health
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Production RAG API",
        description=(
            "Retrieval-Augmented Generation with guardrails, evaluation and observability."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    prefix="/api/v1"
    app.include_router(query.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)
    app.include_router(evaluation.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)

    return app


app = create_app()