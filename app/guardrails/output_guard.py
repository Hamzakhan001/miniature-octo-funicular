from __future__ import annotations

import re
import time
from typing import List

from app.core.logging import log
from app.core.models import DocumentChunk, GuardrailAction, GuardrailResult
from app.guardrails.input_guard import _PII_PATTERNS
from app.observability.tracer import GUARDRAIL_BLOCKS, traced_span
from config.settings import get_settings

_REFUSAL_SIGNALS = re.compile(
    r"\b(i can't help with that|i cannot help with that|i'm sorry, but i can't)\b",
    re.IGNORECASE,
)


class OutputGuard:
    def __init__(self) -> None:
        self._settings = get_settings()

    def check(
        self,
        answer: str,
        sources: List[DocumentChunk],
        run_hallucination_check: bool = True,
    ) -> GuardrailResult:
        t0 = time.perf_counter()
        with traced_span("guardrail.output"):
            result = self._run_checks(answer, sources, run_hallucination_check)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        if result.action != GuardrailAction.ALLOW:
            GUARDRAIL_BLOCKS.labels(stage="output", reason=result.reason or "unknown").inc()
        return result

    def _run_checks(
        self,
        answer: str,
        sources: List[DocumentChunk],
        run_hallucination_check: bool,
    ) -> GuardrailResult:
        if _REFUSAL_SIGNALS.search(answer):
            return GuardrailResult(action=GuardrailAction.BLOCK, reason="refusal_signal")

        if len(answer) > self._settings.output_max_chars:
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                reason="output_too_long",
                redacted_text=answer[: self._settings.output_max_chars],
            )

        if run_hallucination_check and sources:
            hallucination_result = self._check_hallucination(answer, sources)
            if hallucination_result is not None:
                return hallucination_result

        scrubbed, found = self._scrub_pii(answer)
        if found:
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                reason="pii_detected",
                redacted_text=scrubbed,
            )

        return GuardrailResult(action=GuardrailAction.ALLOW, reason="all_checks_passed")

    def _check_hallucination(
        self,
        answer: str,
        sources: List[DocumentChunk],
    ) -> GuardrailResult | None:
        source_text = " ".join(chunk.text.lower() for chunk in sources[:5])
        answer_tokens = {token.lower() for token in re.findall(r"\b\w+\b", answer)}
        if not answer_tokens:
            return None

        grounded = sum(1 for token in answer_tokens if token in source_text)
        grounded_ratio = grounded / len(answer_tokens)
        if grounded_ratio < self._settings.hallucination_threshold:
            log.warning(
                "output_guardrail_hallucination_detected",
                grounded_ratio=round(grounded_ratio, 3),
            )
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="hallucination_detected",
            )
        return None

    def _scrub_pii(self, text: str) -> tuple[str, list[str]]:
        found: list[str] = []
        for name, pattern in _PII_PATTERNS:
            if pattern.search(text):
                found.append(name)
                text = pattern.sub("[REDACTED]", text)
        return text, found
