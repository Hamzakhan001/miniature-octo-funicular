from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def configure_metrics(app: FastAPI) -> None:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app).expose(
        app,
        include_in_schema=False,
        endpoint="/metrics",
        tags=["Observability"],
    )
