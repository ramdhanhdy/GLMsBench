"""Smoke-test a single provider config against a 1-item MMLU run.

Usage:
    ZAI_API_KEY=... UMANS_API_KEY=... python scripts/smoke_test.py --config harness/config.example.yaml

This runs each configured suite at n=1 for both providers, prints the per-request
JSON and confirms the pipeline (client -> runner -> JSON report) works end-to-end
before committing budget to a full run. Keep this in scripts/ — it is not
a unit test (it hits the network and spends tokens).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glmsbench.config import load_config
from glmsbench.runner import run_benchmark
from glmsbench.reporting.json_out import build_report


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="smoke_results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.suite_sizes = {s: 1 for s in cfg.suites}
    cfg.repeats.latency = 1

    records = await run_benchmark(
        cfg, out_dir=args.out, checkpoint_path=str(Path(args.out) / "ck.jsonl"),
        suites=cfg.suites,
    )
    report = build_report(records, {"model": cfg.model, "providers": list(cfg.providers)})
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
