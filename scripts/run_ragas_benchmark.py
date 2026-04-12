from __future__ import annotations

import asyncio
import json
from datetime import datetime,timezone
from pathlib import Path

from app.evals.runner import RagasBenchmarkRunner

DATASET_PATH = Path("data/benchmark_dataset.jsonl")
OUTPUT_DIR = Path("data/eval_reports")

aync def main() -> None:
    runner = RagasBenchmarkRunner(DATASET_PATH)
    result = await runner.run(top_k=5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR/ f"ragas_benchmark_{timestamp}.json"

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")



if __name__ == "__main__":
    asyncio.run(main())


