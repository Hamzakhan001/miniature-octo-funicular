from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional


from app.core.logging import log
router = APIRouter(prefix="/eval", tags=["Evaluation"])


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: List[str]


class EvalResponse(BaseModel):
    faithfulness: Optional[float] = None
    relevance: Optional[float] = None
    note: str = ""
    
@router.post("", response_model= EvalResponse, summary="Evaluate answer quality")
def evaluate(body: EvalRequest):
    try:
        question_words = set(body.question.lower().split())
        answer_words = set(body.answer.lower().split())
        context_text = " ".join(body.context).lower()
        context_words = set(context_text.split())

        relevance = (
            len(question_words & answer_words) / len(question_words) if question_words else 0.0
        )

        faithfulness = (
            len(answer_words & context_words) / len(answer_words) if answer_words else 0.0
        )
        
        return EvalResponse(
            faithfulness=round(faithfulness, 3),
            relevance=round(relevance, 3),
            note="Heuristic evaluation based on word overlap"
        )
    except Exception as e:
        log.error(f"Error evaluating answer: {e}")
        raise e
