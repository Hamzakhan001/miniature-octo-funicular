from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/audit", tags=["Audit"])

AUDIT_LOG_PATH = Path("data/audit_logs.jsonl")


@router.get("")
def get_audit_logs(limit: int=50):
    if not AUDIT_LOG_PATH.exists():
        return {"records": []}

    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines[-limit:]]
    return {"records": records}
