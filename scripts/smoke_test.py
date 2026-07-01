"""Smoke-test a provider config against a tiny (default n=1) benchmark run.

Usage:
    python scripts/smoke_test.py --config harness/config.example.yaml
    python scripts/smoke_test.py --config harness/config.example.yaml --suite arc --n 3

API keys are read from .env (see .env.example) or real env vars.
Runs the selected suite(s) (default: all in config) at n items (default 1)
for both providers, prints the per-request JSON and confirms the pipeline
(client -> runner -> JSON report) works end-to-end before committing budget
to a full run. Keep this in scripts/ — it is not a unit test (it hits the
network and spends tokens).
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
    ap.add_argument("--suite", default=None, help="Comma-list subset of suites (default: all in config)")
    ap.add_argument("--n", type=int, default=1, help="Items per suite (default: 1)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    suites = args.suite.split(",") if args.suite else cfg.suites
    cfg.suite_sizes = {s: args.n for s in suites}
    cfg.repeats.latency = 1

    records = await run_benchmark(
        cfg, out_dir=args.out, checkpoint_path=str(Path(args.out) / "ck.jsonl"),
        suites=suites,
    )
    providers_pricing = {name: p.pricing for name, p in cfg.providers.items()}
    report = build_report(
        records, {"model": cfg.model, "providers": list(cfg.providers)},
        providers_pricing, cfg.usage_tiers,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
