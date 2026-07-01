from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from ..models import RequestRecord
from .aggregates import build_aggregates

if TYPE_CHECKING:
    from ..config import Pricing, UsageTier


def build_report(
    records: list[RequestRecord],
    config_dump: Optional[dict] = None,
    providers_pricing: Optional[dict[str, "Pricing"]] = None,
    usage_tiers: Optional[dict[str, "UsageTier"]] = None,
) -> dict:
    return {
        "run_meta": {
            "config": config_dump or {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_requests": len(records),
        },
        "aggregates": build_aggregates(records, providers_pricing, usage_tiers),
        "requests": [asdict(r) for r in records],
    }


def write_report(
    records: list[RequestRecord],
    path: str,
    config_dump: Optional[dict] = None,
    providers_pricing: Optional[dict[str, "Pricing"]] = None,
    usage_tiers: Optional[dict[str, "UsageTier"]] = None,
):
    report = build_report(records, config_dump, providers_pricing, usage_tiers)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
