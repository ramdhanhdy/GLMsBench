"""GLM-5.2 provider benchmark CLI.

Usage:
    python run.py --config harness/config.example.yaml [--suite mmlu,gsm8k] [--out results/] [--resume]
"""
from __future__ import annotations
import argparse
import asyncio
import os
from datetime import datetime

from glmsbench.config import load_config
from glmsbench.runner import run_benchmark
from glmsbench.reporting.json_out import write_report
from glmsbench.reporting.markdown import build_markdown


async def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 provider benchmark")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--suite", default=None, help="Comma-list subset of suites (default: all)")
    ap.add_argument("--out", default=None, help="Output directory (default: results/<timestamp>/)")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    args = ap.parse_args()

    cfg = load_config(args.config)
    suites = args.suite.split(",") if args.suite else None

    out_dir = args.out or os.path.join("results", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.jsonl")

    if not args.resume and os.path.exists(checkpoint_path):
        # Avoid silently appending to a stale checkpoint from a different run.
        os.remove(checkpoint_path)

    records = await run_benchmark(cfg, out_dir=out_dir, checkpoint_path=checkpoint_path, suites=suites)

    config_dump = {
        "model": cfg.model,
        "providers": list(cfg.providers.keys()),
        "suites": suites or cfg.suites,
        "repeats": {"parity": cfg.repeats.parity, "latency": cfg.repeats.latency},
    }
    write_report(records, os.path.join(out_dir, "results.json"), config_dump)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_markdown(records))

    print(f"Done. {len(records)} records -> {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
