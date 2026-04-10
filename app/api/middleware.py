from __future__ import annotations

import time
import uuid


from starletter.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


from app.core.logging import logger


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID",  str(uuid.uuid4()))
        request.state.request_id = request_id
        logger.bind(request_id=request_id).info("Request started")
        response = await call_next(request)
        logger.bind(request_id=request_id).info("Request completed")
        response.headers["X-Request-ID"] = request_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # TODO: Implement authentication logic
        from config.settings import get_settings

        settings = get_settings()
        skip_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}

        if request.url.path in skip_paths:
            return await call_next(request)

        # TODO: Implement authentication logic
        api_key = request.headers.get(settings.api_key_header)
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
            )
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )
        return await call_next(request)

    
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = round((time.perf_counter() - start_time)*1000, 1)
        logger.bind(
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time=process_time,
        ).info("Request completed")
        return response