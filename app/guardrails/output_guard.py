from __future__ import annotations
import re
import time
from typing import Optional


from app.core.models import GuardrailResult, GuardrailStatus, DocumentChunk
from app.core.logging import log
from app.observability.tracer import GUARDRAIL_BLOCKS, traced_span
from config.settings import get_settings
from app.guardrails.input_guard import _PII_PATTERNS


_REFUSAL_SIGNALS = re.compile(
    range
)

_HALLUCINATION_PROMPT = """\
You are a strict factual auditor

CONTEXT (retrieved documents)
{context}

ANSWER TO AUDIT
{answer}

Task: Decide whether the answer introduces claims that are NOT supported by or that
CONTRADICT the context. Output ONLY  a JSON object:
{{"contradiction_score": <float 0.0-1.0>, "reason":"<one sentence>"}}

contradiction_score: 
                 0.0 means fully grounded; 
                 1.0 means completely fabiricated

"""


class OutputGuard:
    def __init__(self) -> None:
        self._settings = get_settings()

    def guard(
        self,
        answer: str,
        sources: List[DocumentChunk],
        run_hallucination_check: bool = True
    ) -> GuardrailResult:
        t0 = time.perf_counter()
        with traced_span("guardrail.output"):
            result = self._run_checks(answer, sources, run_hallucination_check)
        result.latency_ms = (time.perf_counter() - t0) * 1000

        if result.action != GuardRailAction.ALLOW:
            GUARDRAIL_BLOCKS.labels(stage="output", reason=result.reason or "unkown").inc()

        return result

    def _run_checks(
        self,
        answer: str,
        sources: List[DocumentChunk],
        run_hallucination_check: bool
    ) -> GuardrailResult:
        if _REFUSAL_SIGNALS.search(answer):
            return GuardrailResult(
                action = GuardrailAction.BLOCK,
                reason = "refusal_signal"
            )

        if len(answer) > self._settings.output_max_chars:
            truncated = answer[: self._settings.output_max_chars]
            return GuardrailResult(
                action = GuardrailAction.REDACT,
                reason = "output_too_long",
                redacted_answer = truncated
            )
        if run_hallucination_check and sources:
            hallucination_result = self._check_hallucination(answer, sources)
            if hallucination_result is not None:
                return hallucination_result

        scrubbed, found = self._scrub_pii(answer)
        if found:
            return GuardrailResult(
                action = GuardrailAction.REDACT,
                reason = "pii_detected",
                redacted_answer = scrubbed
            )
        return GuardrailResult(
            action = GuardrailAction.ALLOW,
            reason = "all_checks_passed"
        )

    def _check_hallucination(
        self,
        answer: str,
        sources: List[DocumentChunk]
    ) -> GuardrailResult | None:

    try:
        from app.core.llm_client import LLMClient
        import json

        context = "\n -- \n".json(c.text[:400] for c in sources[:5])
        prompt = _HALLUCINATION_PROMPT.format(context=context, answer=answer)
        client = LLMClient()
        result = client.complete(
            system_prompt = "You are JSON-only responder",
            user_prompt = prompt,
            temperature = 0.0,
            max_tokens = 100,
        )
        raw = result.answer.strip()
        raw = re.sub()
        data = json.loads(raw)
        score = float(data.get("contraction_score", 0.0))
        reason = data.get("reason", "unknown")


        if score > self._settings.hallucination_threshold:
            return GuardrailResult(
                action = GuardrailAction.BLOCK,
                reason = "hallucination_detected",
                hallucination_score = score,
                hallucination_reason = reason
            )
        return None
    
    except Exception as e
        logger.warning(f"Hallucination check failed: {e}")
        return None

    def _scrub_pii(self, text: str) -> tuple[str, list[str]]:
        found: list[str] = []
        for name, pattern in _PII_PATTERNS:
            if pattern.search(text):
                found.append(name)
                text = pattern.sub("[REDACTED]", text)
        return text, found