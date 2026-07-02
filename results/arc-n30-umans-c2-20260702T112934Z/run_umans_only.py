from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from glmsbench.config import load_config
from glmsbench.runner import run_benchmark
from glmsbench.reporting.json_out import build_report


async def main() -> int:
    run_dir = Path(__file__).resolve().parent
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(ROOT / "harness" / "config.example.yaml"))
    cfg.providers = {"umans": cfg.providers["umans"]}
    cfg.providers["umans"].rate.concurrency = 2
    cfg.suites = ["arc"]
    cfg.suite_sizes = {"arc": 30}
    cfg.repeats.parity = 1
    cfg.repeats.latency = 1

    print(json.dumps({
        "run_dir": str(run_dir),
        "provider": "umans",
        "suite": "arc",
        "n_items": 30,
        "requests_expected": 60,
        "concurrency": cfg.providers["umans"].rate.concurrency,
        "latency_repeats": cfg.repeats.latency,
        "parity_repeats": cfg.repeats.parity,
    }, indent=2), flush=True)

    records = await run_benchmark(
        cfg,
        out_dir=str(raw_dir),
        checkpoint_path=str(raw_dir / "ck.jsonl"),
        suites=["arc"],
    )

    report = build_report(
        records,
        {"model": cfg.model, "providers": list(cfg.providers)},
        {name: p.pricing for name, p in cfg.providers.items()},
        cfg.usage_tiers,
    )
    results_path = raw_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    ok = sum(1 for r in records if r.error is None and r.http_status == 200 and r.timing)
    err = len(records) - ok
    throttles = sum((r.rate.n_throttled if r.rate else 0) for r in records)
    print(json.dumps({
        "results": str(results_path),
        "records": len(records),
        "ok": ok,
        "errors": err,
        "throttled_events": throttles,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
