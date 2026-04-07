"""
Input Guardrails - run Before retrieval


Checks:
Length Limit
Blocked Topic Detection
PII detection and redaction using regex patterns
Prompt injection detection

Returns:
GuardrailResult(action=ALLOW|BLOCK|REDACT, reason_redacted_text)

"""

from __future__ import annotations
import re
import time
from typing import Optional

from app.core.models import GuardrailResult, GuardrailAction
from app.core.logging import log
from app.observability.tracer import GUARDRAIL_BLOCKS, traced_span
from config.settings import get_settings

