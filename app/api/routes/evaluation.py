from __future__ import annotations

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.logging import log
from app.services.evaluation import EvaluationService

router = APIRouter(prefix="/eval", tags=["Evaluation"])
evaluator = EvaluationService()


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: List[str]


class EvalResponse(BaseModel):
    faithfulness: float
    relevance: float
    context_coverage: float
    note: str


@router.post("", response_model=EvalResponse, summary="Evaluate answer quality")
def evaluate(body: EvalRequest) -> EvalResponse:
    try:
        result = evaluator.evaluate(
            question=body.question,
            answer=body.answer,
            context=body.context,
        )
        return EvalResponse(
            faithfulness=round(result.faithfulness, 3),
            relevance=round(result.relevance, 3),
            context_coverage=round(result.context_coverage, 3),
            note=result.note,
        )
    except Exception as exc:
        log.exception("evaluation_failed", error=str(exc))
        raise
