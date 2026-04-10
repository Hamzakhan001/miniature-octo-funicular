from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config.settings import get_settings


def configure_tracing(service_name: str = "retrieval-process-docs") -> None:
    settings = get_settings()
    if not settings.enable_tracing:
        return

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": settings.app_env,
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
