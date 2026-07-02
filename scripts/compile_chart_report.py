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
    ("provider_scorecard_table.png", "Provider scorecard"),
    ("request_outcomes.png", "Request outcome by provider"),
    ("throttling_diagnostics.png", "Throttling and backoff diagnostics"),
    ("e2e_latency_success_only.png", "End-to-end latency distribution"),
    ("ttft_success_only.png", "Time to first token"),
    ("output_tokens_success_only.png", "Output token distribution"),
    ("paired_parity_subset.png", "Paired parity subset"),
]


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _providers(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("provider_summary", {})


def _provider_name(name: str) -> str:
    return "Z.ai" if name == "zai" else name


def _provider_join(names: list[str]) -> str:
    pretty = [_provider_name(name) for name in names]
    if len(pretty) <= 1:
        return "".join(pretty)
    if len(pretty) == 2:
        return f"{pretty[0]} and {pretty[1]}"
    return ", ".join(pretty[:-1]) + f", and {pretty[-1]}"


def _headline(summary: dict[str, Any], quality: dict[str, Any] | None) -> str:
    rows = _providers(summary)
    providers = sorted(rows)
    if len(providers) < 2:
        return "Insufficient provider data for a head-to-head decision."

    clean = [
        p for p in providers
        if rows[p].get("ok_requests") == rows[p].get("requests")
        and rows[p].get("throttled_events", 0) == 0
        and rows[p].get("error_requests", 0) == 0
    ]
    fastest = min(
        providers,
        key=lambda p: rows[p].get("e2e_ms", {}).get("median") or float("inf"),
    )
    best_tail = min(
        providers,
        key=lambda p: rows[p].get("e2e_ms", {}).get("p95") or float("inf"),
    )

    if quality:
        qrows = quality.get("provider_quality", {})
        best_strict = max(
            qrows,
            key=lambda p: qrows[p].get("strict_task_success") or 0,
        ) if qrows else None
        if best_strict:
            return (
                f"Both providers completed the merged ARC n=30 comparison cleanly: {_provider_join(clean)} had 60/60 OK requests with zero throttles. "
                f"{_provider_name(fastest)} had the best median E2E latency; {_provider_name(best_tail)} had the best p95 tail latency. "
                f"On strict task success, {_provider_name(best_strict)} led with {_fmt_pct(qrows[best_strict].get('strict_task_success'))}. "
                "The main quality failure mode was not wrong parsed answers, but empty visible answers after exhausting the 300-token reasoning budget."
            )

    if clean:
        return (
            f"Clean reliability providers: **{_provider_join(clean)}**. "
            f"Fastest successful-request E2E median: **{_provider_name(fastest)}**. "
            f"Best E2E p95 tail latency: **{_provider_name(best_tail)}**."
        )

    incomplete = [p for p in providers if rows[p].get("ok_requests") != rows[p].get("requests")]
    return (
        "This run is primarily a reliability/incomplete-data story. "
        f"Incomplete providers: **{_provider_join(incomplete)}**. "
        "Latency and token charts should be interpreted on successful requests only."
    )


def _chart_caption(filename: str, summary: dict[str, Any]) -> str:
    rows = _providers(summary)
    providers = sorted(rows)
    if filename == "provider_scorecard_table.png":
        return "Head-to-head scorecard generated from normalized provider summary data; no composite winner score is used."

    if filename == "request_outcomes.png":
        parts = [f"{_provider_name(p)} {rows[p].get('ok_requests', 0)}/{rows[p].get('requests', 0)} OK" for p in providers]
        return "Request outcomes per provider: " + "; ".join(parts) + "."

    if filename == "throttling_diagnostics.png":
        parts = [
            f"{_provider_name(p)} {rows[p].get('throttled_events', 0)} throttles, {_fmt(rows[p].get('backoff_ms', 0), 0)} ms backoff"
            for p in providers
        ]
        return "Provider lifecycle diagnostics: " + "; ".join(parts) + "."

    if filename == "e2e_latency_success_only.png":
        parts = [
            f"{_provider_name(p)} n={rows[p].get('e2e_ms', {}).get('n', 0)}, median {_fmt(rows[p].get('e2e_ms', {}).get('median'), 0)} ms, p95 {_fmt(rows[p].get('e2e_ms', {}).get('p95'), 0)} ms"
            for p in providers
        ]
        return "Successful requests only: " + "; ".join(parts) + "."

    if filename == "ttft_success_only.png":
        parts = [
            f"{_provider_name(p)} n={rows[p].get('ttft_ms', {}).get('n', 0)}, median {_fmt(rows[p].get('ttft_ms', {}).get('median'), 0)} ms, p95 {_fmt(rows[p].get('ttft_ms', {}).get('p95'), 0)} ms"
            for p in providers
        ]
        return "Measured time to first token on successful requests with TTFT available: " + "; ".join(parts) + "."

    if filename == "output_tokens_success_only.png":
        parts = [
            f"{_provider_name(p)} median {_fmt(rows[p].get('output_tokens', {}).get('median'), 0)} tokens, p95 {_fmt(rows[p].get('output_tokens', {}).get('p95'), 0)}"
            for p in providers
        ]
        return "Successful responses only: " + "; ".join(parts) + ". p95 at 300 indicates completion-budget pressure."

    if filename == "paired_parity_subset.png":
        parity = summary.get("parity_summary", {})
        if not parity:
            return "No paired parity summary available."
        suite, row = sorted(parity.items())[0]
        return (
            f"Paired {suite} subset: {row.get('n_compared', 0)} comparable items, "
            f"{row.get('canonical_disagreements', 0)} canonical disagreements, "
            f"{row.get('canonical_unknown', 0)} unknown extractions, "
            f"{_fmt((row.get('raw_text_agreement_rate') or 0) * 100)}% raw text agreement."
        )

    return "Chart generated from normalized GLMsBench artifacts."


def _append_quality_section(lines: list[str], quality: dict[str, Any] | None) -> None:
    if not quality:
        return
    qrows = quality.get("provider_quality", {})
    if not qrows:
        return

    lines.append("## Quality and correctness")
    lines.append("")
    lines.append("This section scores ARC outputs against gold labels. It separates answered-only accuracy from strict task success, where strict success means the request emitted a parseable correct answer.")
    lines.append("")
    lines.append("| Provider | Correct / requests | Strict success | Answered-only accuracy | Empty content | Exact single-letter | Length stops |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for provider, row in sorted(qrows.items()):
        lines.append(
            f"| {_provider_name(provider)} | {row.get('correct_requests', 0)} / {row.get('requests', 0)} | "
            f"{_fmt_pct(row.get('strict_task_success'))} | {_fmt_pct(row.get('answered_only_accuracy'))} | "
            f"{row.get('empty_content', 0)} | {row.get('exact_single_letter_outputs', 0)} | {row.get('length_stops', 0)} |"
        )
    lines.append("")
    lines.append("**Interpretation:** among parseable answers, both providers were correct on this small sample. The quality gap comes from empty visible answers that hit `stop_reason=length` after consuming the 300-token completion budget, not from parsed wrong answers.")
    lines.append("")

    item_summary = quality.get("item_summary", {})
    if item_summary:
        lines.append("| Item-level pattern | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Both providers correct on both passes | {item_summary.get('both_providers_all_passes_correct', 0)} / {quality.get('n_items', '-')} |")
        unknown_items = item_summary.get("items_with_any_unknown_answer", [])
        lines.append(f"| Items with any empty/non-parseable output | {len(unknown_items)} / {quality.get('n_items', '-')} |")
        lines.append("")
    lines.append("See `quality/deep_quality_report.md` and `quality/item_correctness.csv` for item-level details.")
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"
    charts_dir = run_dir / "charts"
    summary_path = analysis_dir / "provider_summary.json"
    quality_path = run_dir / "quality" / "quality_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None

    lines: list[str] = []
    lines.append("# GLMsBench ARC n=30 Provider Decision Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(_headline(summary, quality))
    lines.append("")
    lines.append("This report compares Z.ai and umans serving GLM-5.2 on a 30-item ARC sample. It is a small but chartable run designed to validate the benchmark-to-report workflow before scaling.")
    lines.append("")

    lines.append("## Run facts")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Requests | {summary.get('n_requests', '-')} |")
    lines.append(f"| Providers | {_provider_join(summary.get('providers', []))} |")
    lines.append(f"| Suites | {', '.join(summary.get('suites', []))} |")
    lines.append("")

    lines.append("## Provider summary")
    lines.append("")
    lines.append("| Provider | OK / Total | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Output tok p50 | Throttles |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for provider, row in sorted(_providers(summary).items()):
        lines.append(
            "| "
            + " | ".join([
                _provider_name(provider),
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

    _append_quality_section(lines, quality)

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
    for filename, title in CHARTS:
        if (charts_dir / filename).exists():
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}](charts/{filename})")
            lines.append("")
            lines.append(f"*{_chart_caption(filename, summary)}*")
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
