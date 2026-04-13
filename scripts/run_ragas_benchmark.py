from __future__ import annotations

import asyncio
import json
from datetime import datetime,timezone
from pathlib import Path

from app.evals.runner import RagasBenchmarkRunner

DATASET_PATH = Path("data/benchmark_seed.jsonl")
OUTPUT_DIR = Path("data/eval_reports")

async def main() -> None:
    runner = RagasBenchmarkRunner(DATASET_PATH)
    result = await runner.run(top_k=5)

    faithfulness = result["ragas"]["faithfulness"]
    answer_relevancy = result["ragas"]["answer_relevancy"]
    context_recall = result["ragas"]["context_recall"]

    if faithfulness < 0.70:
        raise SystemExit("Faithfulness score is below 0.70")
    
    if answer_relevancy < 0.60:
        raise SystemExit("Answer relevancy score is below 0.60")
    
    if context_recall < 0.70:
        raise SystemExit("Context recall score is below 0.70")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR/ f"ragas_benchmark_{timestamp}.json"

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")



if __name__ == "__main__":
    asyncio.run(main())


