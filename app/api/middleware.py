from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import (
    bind_request_context,
    clear_request_context,
    logger,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        bind_request_context(request_id=request_id)

        try:
            response = await call_next(request)
        finally:
            clear_request_context()

        response.headers["X-Request-ID"] = request_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        from config.settings import get_settings

        settings = get_settings()
        skip_paths = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
        }

        if request.url.path in skip_paths:
            return await call_next(request)

        api_keys = settings.api_keys or []
        if not api_keys:
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header)
        if not api_key:
            return JSONResponse(status_code=401, content={"detail": "Missing API key"})
        if api_key not in api_keys:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 1)
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                request_id=getattr(request.state, "request_id", None),
                latency_ms=process_time_ms,
            )
            raise

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 1)
        logger.info(
            "request_completed",
            request_id=getattr(request.state, "request_id", None),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=process_time_ms,
        )
        return response
