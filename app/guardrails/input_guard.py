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

_PII_PATTERNS : list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone_us",    re.compile(r"\b(\+1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")),
    ("ssn",         re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
    ("api_key",     re.compile(r"\b(sk|pk|api)[_\-][a-zA-Z0-9]{20,}\b")),
]

_INJECTION_SIGNALS = [
    r"ignore (all |previous |above )?instructions",
    r"forget (everything|all|your|what you)",
    r"you are now (a |an )?(?!assistant)",
    r"(pretend|act|behave) (you are|as if|like) (a |an )?(?!assistant)",
    r"disregard (your|all|safety|guidelines)",
    r"jailbreak",
    r"developer mode",
    r"DAN\b",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_SIGNALS), re.IGNORECASE)

class InputGuard:
    def __init__(self):
        self._settings = get_settings()

        def check(self, text: str) -> GuardrailResult:
            t0 = time.perf_counter()
            with traced_span("guardrail.input"):
                result = self._run_checks(text)
                
            result.latency_ms = (time.perf_counter() - t0) * 1000

            if result.action != GuardrailAction.ALLOW:
                log.warning("Input guardrail blocked request", extra={"guardrail": "input", "action": result.action.value, "reason": result.reason})
            
                GUARDTAIL_BLOCKS.labels(stage="input", reason = result.reason or "unknown").inc()
            return result

        def _run_checks(self, text: str) -> GuardrailResult:
            if len(text) > self._settings.input_max_chars:
                return GuardrailResult(
                    action = GuardrailAction.BLOCK,
                    reason = "input_too_long",
                )
            
            if _INJECTION_RE.search(text):
                return GuardrailResult(
                    action = GuardrailAction.BLOCK,
                    reason = "Potential Prompt Injection Detected"
                )

            lower = text.lower()
            for topic in self._settings.blocked_topics:
                if topic.lower() in lower:
                    return GuardrailResult(
                        action = GuardrailAction.BLOCK,
                        reason = "blocked_topic"
                    )
            
            if self._settings.pii_detection:
                redacted, found = self._redact_pii(text)
                if found:
                    return GuardrailResult(
                        action = GuardrailAction.REDACT,
                        reason = "pii_detected",
                        redacted_text = redacted
                    )
                
            return GuardrailResult(
                action = GuardrailAction.ALLOW,
                reason = "allowed"
            )

        def _redact_pii(self, text: str) -> tuple[str, list[str]]:
            found: list[str] = []
            for name, pattern in _PII_PATTERNS:
                if pattern.search(text):
                    found.append(name)
                    text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
            return text, found
            