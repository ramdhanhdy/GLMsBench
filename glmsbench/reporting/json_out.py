from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from ..models import RequestRecord


def build_report(records: list[RequestRecord], config_dump: Optional[dict] = None) -> dict:
    return {
        "run_meta": {
            "config": config_dump or {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_requests": len(records),
        },
        "requests": [asdict(r) for r in records],
    }


def write_report(records: list[RequestRecord], path: str, config_dump: Optional[dict] = None):
    report = build_report(records, config_dump)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
