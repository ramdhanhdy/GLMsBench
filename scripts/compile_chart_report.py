"""Compile a chart-centered Markdown report for a GLMsBench run.

Usage:
    python3 scripts/compile_chart_report.py \
      --run-dir results/arc-n30-20260701T120000Z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHARTS = [
    (
        "provider_scorecard_table.png",
        "Provider scorecard",
        "Head-to-head scorecard. Latency and token stats are computed on successful requests only; no composite winner score is used.",
    ),
    (
        "request_outcomes.png",
        "Request outcome by provider",
        "Request outcomes per provider, n=60 each. Z.ai returned 60/60 OK; umans returned 14/60 OK and 46 errors.",
    ),
    (
        "throttling_diagnostics.png",
        "Throttling and backoff diagnostics",
        "Provider lifecycle diagnostics. umans saw 16 throttled events and 24.3s cumulative backoff; Z.ai saw none.",
    ),
    (
        "e2e_latency_success_only.png",
        "End-to-end latency distribution",
        "Successful requests only. umans n=14, median 6773 ms; Z.ai n=60, median 4860 ms.",
    ),
    (
        "ttft_success_only.png",
        "Time to first token",
        "Measured successful requests only. umans n=11, median 6307 ms; Z.ai n=52, median 4652 ms.",
    ),
    (
        "output_tokens_success_only.png",
        "Output token distribution",
        "Successful responses only. umans median 212.5 tokens; Z.ai median 154 tokens. p95=300 for both, likely from a cap.",
    ),
    (
        "paired_parity_subset.png",
        "Paired parity subset",
        "Paired subset only. 7 comparable items, 0 canonical disagreements, 2 unknown extractions. Not a full-suite agreement rate.",
    ),
]


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _headline(summary: dict[str, Any]) -> str:
    providers = sorted(summary.get("provider_summary", {}))
    if len(providers) < 2:
        return "Insufficient provider data."
    rows = summary["provider_summary"]
    zai = rows.get("zai", {})
    umans = rows.get("umans", {})
    if zai and umans:
        return (
            "This run is primarily a **reliability failure story**, not a clean latency race. "
            f"Z.ai completed {zai.get('ok_requests', 0)}/{zai.get('requests', 0)} requests; "
            f"umans completed {umans.get('ok_requests', 0)}/{umans.get('requests', 0)} and then entered a throttle/suspension failure mode. "
            "Latency and token charts below are therefore explicitly limited to successful requests."
        )
    fastest = min(providers, key=lambda p: rows[p]["e2e_ms"].get("median") or float("inf"))
    reliable = [p for p in providers if rows[p].get("error_requests", 0) == 0 and rows[p].get("throttled_events", 0) == 0]
    reliability_text = ", ".join(reliable) if reliable else "none"
    return (
        f"Fastest successful-request E2E median: **{fastest}**. Reliability clean run: **{reliability_text}**."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"
    charts_dir = run_dir / "charts"
    summary_path = analysis_dir / "provider_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# GLMsBench ARC n=30 Provider Decision Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(_headline(summary))
    lines.append("")
    lines.append("This report compares Z.ai and umans serving GLM-5.2 on a 30-item ARC sample. It is a small but chartable run designed to validate the benchmark-to-report workflow before scaling.")
    lines.append("")

    lines.append("## Run facts")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Requests | {summary.get('n_requests', '-')} |")
    lines.append(f"| Providers | {', '.join(summary.get('providers', []))} |")
    lines.append(f"| Suites | {', '.join(summary.get('suites', []))} |")
    lines.append("")

    lines.append("## Provider summary")
    lines.append("")
    lines.append("| Provider | OK / Total | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Output tok p50 | Throttles |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for provider, row in sorted(summary["provider_summary"].items()):
        lines.append(
            "| "
            + " | ".join([
                provider,
                f"{row['ok_requests']} / {row['requests']}",
                _fmt(row["ttft_ms"].get("median")),
                _fmt(row["ttft_ms"].get("p95")),
                _fmt(row["e2e_ms"].get("median")),
                _fmt(row["e2e_ms"].get("p95")),
                _fmt(row["output_tokens"].get("median"), 0),
                str(row.get("throttled_events", 0)),
            ])
            + " |"
        )
    lines.append("")

    lines.append("## Parity")
    lines.append("")
    lines.append("| Suite | Compared | Canonical disagreements | Unknown | Disagreement rate | Raw text agreement |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite, row in sorted(summary.get("parity_summary", {}).items()):
        lines.append(
            f"| {suite} | {row.get('n_compared', 0)} | {row.get('canonical_disagreements', 0)} | "
            f"{row.get('canonical_unknown', 0)} | {_fmt((row.get('canonical_disagreement_rate') or 0) * 100)}% | "
            f"{_fmt((row.get('raw_text_agreement_rate') or 0) * 100)}% |"
        )
    lines.append("")

    stats = _read_optional(analysis_dir / "stats-findings.md")
    if stats:
        lines.append("## Analyst findings")
        lines.append("")
        lines.append(stats)
        lines.append("")

    chart_spec = _read_optional(analysis_dir / "chart-spec.md")
    if chart_spec:
        lines.append("## Visualization design notes")
        lines.append("")
        lines.append(chart_spec)
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    for filename, title, caption in CHARTS:
        if (charts_dir / filename).exists():
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}](charts/{filename})")
            lines.append("")
            lines.append(f"*{caption}*")
            lines.append("")

    chart_qa = _read_optional(run_dir / "chart-qa-report.md")
    if chart_qa:
        lines.append("## Chart QA")
        lines.append("")
        lines.append(chart_qa)
        lines.append("")

    final_judge = _read_optional(run_dir / "final-judge-report.md")
    if final_judge:
        lines.append("## Final judge")
        lines.append("")
        lines.append(final_judge)
        lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
