from __future__ import annotations

import json
from dataclasses import dataclass,asdict
from pathlib import Path
from statistics import mean
from typing import Any

from app.rag.pipeline import RAGPipeline

@dataclass
class BenchmarkCase:
    id: str
    category: str
    question: str
    reference_answer: str
    reference_context: list[str]
    expected_sources: list[str]
    should_answer: bool
    difficulty: str


@dataclass
class BenchmarkRunRow:
    id: str
    category: str
    difficulty: str
    question: str
    should_answer: bool
    answer: str
    sources: list[str]
    docs_retrieved: int
    latency_ms: float
    eval_scores: dict[str, float] | None
    source_hit: bool
    answered: bool

class RagasBenchmarkRunner:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.pipeline = RAGPipeline()

    def load_cases(self) -> list[BenchmarkCase]:
        cases: list[BenchmarkBase] = []
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(BenchmarkCase(**raw))
        return cases
    
    async def run(self, top_k: int = 5) -> dict[str, Any]:
        cases= self.load_cases()
        rows: list[BenchmarkRunRow] = []
        
        for case in cases:
            response = await self.pipeline.run(
                question = case.question,
                top_k = top_k,
                run_eval = False
            )

            sources_name = [
                source.get("metadata", {}).get("source", "unkown")
                for source in response.sources
            ]

            source_hit = any(src in source_names for src in case.expected_sources) if case.expected_sources else False
            answered = bool(response.answer and "[BLOCKED]" not in response.answer)

            rows.append(
                BenchmarkRunRow(
                    id=case.id,
                    category=case.category,
                    difficulty=case.difficulty,
                    question=case.question,
                    should_answer=case.should_answer,
                    answer=response.answer,
                    sources=source_name,
                    docs_retrieved=len(response.sources),
                    latency_ms=response.latency_ms,
                    eval_scores=response.eval_scores,
                    source_hit=source_hit,
                    answered=answered
                )
            )
        ragas_score = self._run_ragas(cases, rows)

        summary= {
            "total_cases":len(rows),
            "avg_latency_ms": mean(row.latency_ms for row in rows) if rows else 0.0
            "avg_docs_retrieved": mean(row.docs_retrieved for row in rows) if rows else 0.0,
            "source_hit_rate": (
                sum(1 for row in rows if row.source_hit)/len(rows) if rows else 0.0
            ),
            "answer_rate": (
                sum(1 for row in rows if row.answered)/len(rows) if rows else 0.0
            ),
            "ragas": ragas_score,
        }

        def _run_ragas(
            self,
            cases: list[BenchmarkCase],
            rows: list[BenchmarkRunRow]
        ) -> dict[str, float] | None:
            try:
                from datasets import Dataset
                from ragas import evaluate
                from ragas.metrices import answer_relevancy, context_precision, context_recall
            except ImportError:
                return None

            dataset = Dataset.from_dict({
                "question": [case.question for case in cases],
                "answer": [row.answer for row in rows],
                "contexts": [case.reference_context for case in cases],
                "ground_truths": [case.reference_answer for case in cases]
            })
            result = evaluate(
                dataset,
                metrics= [faithfulness, answer_relevancy, context_recall]
            )

            result_dict = result.to_pandas().mean(numeric_only=True).to_dict()

            return {
                "faithfulness": float(result_dict.get("faithfulness", 0.0)),
                "answer_relevancy": float(result_dict.get("answer_relevancy", 0.0)),
                "context_recall": float(result_dict.get("context_recall", 0.0)),
            }



