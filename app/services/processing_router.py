from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


class ProcessingRouter:
    """Decides whether a document should stay on the Lambda path or be offloaded to Fargate."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._lambda_supported_extensions = {
            ext.lower() for ext in self.settings.lambda_supported_extensions
        }

        self._fargate_preferred_extensions = {
            ext.lower() for ext in self.settings.fargate_preferred_extensions
        }

    def pick_target(self, *, filename: str, file_size_bytes: int) -> tuple[str, str]:
        suffix = Path(filename).suffix.lower()
        lambda_limit_bytes = self.settings.lambda_max_inline_file_size_mb * 1024 * 1024

        if suffix in self._fargate_preferred_extensions:
            return "fargate", f"{suffix} files are configured for the heavier processing path"

        if file_size_bytes > lambda_limit_bytes:
            return "fargate", "file size exceeds the Lambda inline threshold"

        if self._lambda_supported_extensions and suffix not in self._lambda_supported_extensions:
            return "fargate", "file type is not in the Lambda-optimized allowlist"

        return "lambda", "file is small and on the Lambda-optimized path"
