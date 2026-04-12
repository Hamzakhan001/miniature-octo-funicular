from __future__ import annotations

import json
from pathlib import Path

from app.core.logging import logger

AUDIT_LOG_PATH= Path("data/audit_logs.jsonl")


def write_audit_record(record: dict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents= True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("audit_record_written", event_type=record.get("event_type"))
