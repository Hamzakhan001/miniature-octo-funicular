from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass(slots=True)
class EvaluationResult:
    faithfulness: float
    relevance: float
    context_coverage: float
    note: str

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "context_coverage": round(self.context_coverage, 3),
        }


class EvaluationService:
    """
    Lightweight lexical evaluation.

    This is still heuristic, but it is much safer than raw word overlap:
    - normalizes tokens
    - ignores repeated-token inflation
    - reports context coverage separately
    """

    def evaluate(self, question: str, answer: str, context: Iterable[str]) -> EvaluationResult:
        question_tokens = self._tokenize(question)
        answer_tokens = self._tokenize(answer)
        context_tokens = self._tokenize(" ".join(context))

        question_set = set(question_tokens)
        answer_set = set(answer_tokens)
        context_set = set(context_tokens)

        relevance = self._safe_ratio(len(question_set & answer_set), len(question_set))
        faithfulness = self._safe_ratio(len(answer_set & context_set), len(answer_set))

        overlap_counter = Counter(answer_tokens)
        grounded_hits = sum(count for token, count in overlap_counter.items() if token in context_set)
        context_coverage = self._safe_ratio(grounded_hits, len(answer_tokens))

        return EvaluationResult(
            faithfulness=faithfulness,
            relevance=relevance,
            context_coverage=context_coverage,
            note=(
                "Lexical grounding heuristic. Good for smoke tests and regressions, "
                "not a replacement for judge-model or human evals."
            ),
        )

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator
