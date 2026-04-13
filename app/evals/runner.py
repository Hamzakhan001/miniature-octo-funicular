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
    reference_contexts: list[str]
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
        cases: list[BenchmarkCase] = []
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(BenchmarkCase(**raw))
        return cases
    
    async def run(self, top_k: int = 3) -> dict[str, Any]:
        cases= self.load_cases()
        rows: list[BenchmarkRunRow] = []
        
        print(f"Loaded {len(cases)} cases")
        
        for i, case in enumerate(cases):
            try:
                print(f"Processing case {i+1}: {case.question[:50]}...")
                response = await self.pipeline.run(
                    question = case.question,
                    top_k = top_k,
                    run_eval = False
                )
            except Exception as e:
                print(f"Error processing case {i+1}: {e}")
                continue

            sources_names = [
                source.get("metadata", {}).get("source", "unkown")
                for source in response.sources
            ]

            source_hit = any(src in sources_names for src in case.expected_sources) if case.expected_sources else False
            answered = bool(response.answer and "[BLOCKED]" not in response.answer)

            rows.append(
                BenchmarkRunRow(
                    id=case.id,
                    category=case.category,
                    difficulty=case.difficulty,
                    question=case.question,
                    should_answer=case.should_answer,
                    answer=response.answer,
                    sources=sources_names,
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
            "avg_latency_ms": mean(row.latency_ms for row in rows) if rows else 0.0,
            "avg_docs_retrieved": mean(row.docs_retrieved for row in rows) if rows else 0.0,
            "source_hit_rate": (
                sum(1 for row in rows if row.source_hit)/len(rows) if rows else 0.0
            ),
            "answer_rate": (
                sum(1 for row in rows if row.answered)/len(rows) if rows else 0.0
            ),
            "ragas": ragas_score,
        }
        
        return summary

    def _run_ragas(
        self,
        cases: list[BenchmarkCase],
        rows: list[BenchmarkRunRow]
    ) -> dict[str, float] | None:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            from config.settings import get_settings
            import os
            
            # Set OpenAI API key for RAGAS
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

            judge_llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )

            judge_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small")
            )

            faithfulness_metric = Faithfulness(llm=judge_llm)
            answer_relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
            context_recall_metric = ContextRecall(llm=judge_llm)

        except Exception as exc:
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc


        dataset = Dataset.from_dict({
            "question": [case.question for case in cases],
            "answer": [row.answer for row in rows],
            "contexts": [case.reference_contexts for case in cases],
            "reference": [case.reference_answer for case in cases]
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric, context_recall_metric],
        )

        df = result.to_pandas()
        print(df.columns.tolist())


        print(df[["user_input", "answer_relevancy"]])
        print(df["answer_relevancy"].isna().sum())
        print(df["answer_relevancy"].notna().sum())




        result_dict = result.to_pandas().mean(numeric_only=True).to_dict()

        import math

        def safe_mean(series) -> float:
            values = [
                v for v in series.tolist()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            return sum(values) / len(values) if values else 0.0

        def valid_count(series) -> int:
            return sum(
                1 for v in series.tolist()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            )

        ragas_scores = {
            "faithfulness": safe_mean(df["faithfulness"]),
            "answer_relevancy": safe_mean(df["answer_relevancy"]),
            "context_recall": safe_mean(df["context_recall"]),
            "meta": {
                "faithfulness_valid_rows": valid_count(df["faithfulness"]),
                "answer_relevancy_valid_rows": valid_count(df["answer_relevancy"]),
                "context_recall_valid_rows": valid_count(df["context_recall"]),
            },
        }

        print(ragas_scores)



        return {
            "faithfulness": float(result_dict.get("faithfulness", 0.0)),
            "answer_relevancy": float(result_dict.get("answer_relevancy", 0.0)),
            "context_recall": float(result_dict.get("context_recall", 0.0)),
        }



