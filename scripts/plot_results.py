"""Generate GLMsBench chart images from normalized CSV/JSON artifacts.

Usage:
    python3 scripts/plot_results.py \
      --analysis results/arc-n30/analysis \
      --charts results/arc-n30/charts
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "zai": "#4f46e5",
    "umans": "#16a34a",
}
ERROR_COLOR = "#dc2626"
UNKNOWN_COLOR = "#f59e0b"


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _fmt(value: Any, digits: int = 1) -> str:
    x = _num(value)
    if x is None:
        return "-"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _provider_values(rows: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("ok", "")).lower() != "true":
            continue
        provider = str(row["provider"])
        value = _num(row.get(key))
        if value is None:
            continue
        out.setdefault(provider, []).append(value)
    return out


def _metric(summary: dict[str, Any], provider: str, metric: str, stat: str) -> float | None:
    return summary["provider_summary"][provider][metric].get(stat)


def _providers(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["provider_summary"])


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _annotate_box_counts(ax, providers: list[str], values: dict[str, list[float]], summary: dict[str, Any], metric: str) -> None:
    ymax = max((max(v) for v in values.values() if v), default=1.0)
    for idx, provider in enumerate(providers, start=1):
        n = len(values.get(provider, []))
        median = _metric(summary, provider, metric, "median")
        p95 = _metric(summary, provider, metric, "p95")
        ax.text(idx, ymax * 1.03, f"n={n}\nmed={_fmt(median)}\np95={_fmt(p95)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=ymax * 1.22)


def provider_scorecard_table(summary: dict[str, Any], charts: Path) -> None:
    providers = _providers(summary)
    parity = next(iter(summary.get("parity_summary", {}).values()), {})
    rows: list[list[str]] = []
    for provider in providers:
        p = summary["provider_summary"][provider]
        rows.append([
            provider,
            f"{p.get('requests', 0)}",
            f"{p.get('ok_requests', 0)}",
            f"{p.get('error_requests', 0)}",
            f"{p.get('throttled_events', 0)}",
            _fmt(p.get("backoff_ms")),
            _fmt(p.get("avg_attempts")),
            _fmt(p["ttft_ms"].get("median")),
            _fmt(p["e2e_ms"].get("median")),
            _fmt(p["output_tokens"].get("median")),
            f"paired n={parity.get('n_compared', 0)}, canon Δ={parity.get('canonical_disagreements', 0)}",
        ])
    columns = ["Provider", "Req", "OK", "Err", "Throt", "Backoff ms", "Avg att", "TTFT med", "E2E med", "Out tok med", "Parity note"]
    fig, ax = plt.subplots(figsize=(16.5, 2.9 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title("Provider scorecard (no composite score)", loc="left", fontsize=14, fontweight="bold", pad=16)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.085, 0.055, 0.055, 0.055, 0.065, 0.085, 0.07, 0.08, 0.08, 0.09, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif rows[r - 1][0] == "zai":
            cell.set_facecolor("#eef2ff")
        elif rows[r - 1][0] == "umans":
            cell.set_facecolor("#f0fdf4")
    _save(fig, charts / "provider_scorecard_table.png")


def request_outcomes(summary: dict[str, Any], charts: Path) -> None:
    providers = _providers(summary)
    ok = [summary["provider_summary"][p].get("ok_requests", 0) for p in providers]
    errors = [summary["provider_summary"][p].get("error_requests", 0) for p in providers]
    y = list(range(len(providers)))
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    ax.barh(y, ok, color="#16a34a", label="OK")
    ax.barh(y, errors, left=ok, color=ERROR_COLOR, label="Error")
    ax.set_yticks(y)
    ax.set_yticklabels(providers)
    ax.set_xlim(0, max([60, *[o + e for o, e in zip(ok, errors)]]) + 2)
    ax.set_xlabel("Requests")
    ax.set_title("Request outcome by provider (n=60 each)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for i, (o, e) in enumerate(zip(ok, errors)):
        ax.text(max(o / 2, 0.7), i, str(o), va="center", ha="center", color="white", fontweight="bold")
        if e:
            ax.text(o + e / 2, i, str(e), va="center", ha="center", color="white", fontweight="bold")
    _save(fig, charts / "request_outcomes.png")


def throttling_diagnostics(summary: dict[str, Any], charts: Path) -> None:
    providers = _providers(summary)
    rows = []
    for provider in providers:
        p = summary["provider_summary"][provider]
        rows.append([provider, str(p.get("throttled_events", 0)), _fmt(p.get("backoff_ms")), _fmt(p.get("avg_attempts"))])
    fig, ax = plt.subplots(figsize=(7.5, 2.8 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title("Throttling and backoff diagnostics", loc="left", fontsize=14, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=["Provider", "Throttle events", "Backoff ms", "Avg attempts"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif rows[r - 1][0] == "umans":
            cell.set_facecolor("#fef2f2")
        else:
            cell.set_facecolor("#f8fafc")
    _save(fig, charts / "throttling_diagnostics.png")


def latency_boxplot(rows: list[dict[str, Any]], summary: dict[str, Any], charts: Path) -> None:
    values = _provider_values(rows, "e2e_ms")
    providers = _providers(summary)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bp = ax.boxplot([values.get(p, []) for p in providers], tick_labels=providers, patch_artist=True, showmeans=True)
    for patch, provider in zip(bp["boxes"], providers):
        patch.set_facecolor(COLORS.get(provider, "#64748b"))
        patch.set_alpha(0.65)
    for idx, provider in enumerate(providers, start=1):
        jitter = [-0.06, -0.03, 0, 0.03, 0.06]
        for j, value in enumerate(values.get(provider, [])):
            ax.plot(idx + jitter[j % len(jitter)], value, "o", color=COLORS.get(provider, "#64748b"), alpha=0.45, markersize=3)
    _annotate_box_counts(ax, providers, values, summary, "e2e_ms")
    ax.set_title("End-to-end latency distribution (successful requests only)")
    ax.set_ylabel("E2E latency (ms)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, charts / "e2e_latency_success_only.png")


def ttft_p50_p95(summary: dict[str, Any], charts: Path) -> None:
    providers = _providers(summary)
    p50 = [_metric(summary, p, "ttft_ms", "median") or 0 for p in providers]
    p95 = [_metric(summary, p, "ttft_ms", "p95") or 0 for p in providers]
    n = [summary["provider_summary"][p]["ttft_ms"].get("n", 0) for p in providers]
    x = list(range(len(providers)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars1 = ax.bar([i - width / 2 for i in x], p50, width, label="median", color="#60a5fa")
    bars2 = ax.bar([i + width / 2 for i in x], p95, width, label="p95", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nn={n_i}" for p, n_i in zip(providers, n)])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to first token (measured successful requests only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars1, labels=[_fmt(v) for v in p50], fontsize=8, padding=2)
    ax.bar_label(bars2, labels=[_fmt(v) for v in p95], fontsize=8, padding=2)
    _save(fig, charts / "ttft_success_only.png")


def output_tokens_boxplot(rows: list[dict[str, Any]], summary: dict[str, Any], charts: Path) -> None:
    values = _provider_values(rows, "output_tokens")
    providers = _providers(summary)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bp = ax.boxplot([values.get(p, []) for p in providers], tick_labels=providers, patch_artist=True, showmeans=True)
    for patch, provider in zip(bp["boxes"], providers):
        patch.set_facecolor(COLORS.get(provider, "#64748b"))
        patch.set_alpha(0.65)
    ax.set_xticklabels([
        f"{provider}\nn={len(values.get(provider, []))}, med={_fmt(_metric(summary, provider, 'output_tokens', 'median'))}"
        for provider in providers
    ])
    ax.set_title("Output tokens per successful request", pad=18)
    ax.set_ylabel("Output tokens")
    ymax = max((max(v) for v in values.values() if v), default=1.0)
    ax.set_ylim(top=ymax * 1.12)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, charts / "output_tokens_success_only.png")


def parity_table(per_item_rows: list[dict[str, Any]], charts: Path) -> None:
    rows = []
    colors = []
    for row in per_item_rows:
        unknown = str(row.get("canonical_unknown", "")).lower() == "true"
        agree = str(row.get("canonical_agree", "")).lower() == "true"
        rows.append([
            row.get("item_id", ""),
            row.get("umans_answer", "") or "unknown",
            row.get("zai_answer", "") or "unknown",
            "yes" if agree else "no",
            "yes" if unknown else "no",
        ])
        colors.append("#fffbeb" if unknown else "#ecfdf5")
    fig, ax = plt.subplots(figsize=(8.5, 2.6 + 0.25 * len(rows)))
    ax.axis("off")
    ax.set_title("Paired parity subset (7 comparable items)", loc="left", fontsize=14, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=["Item", "umans", "zai", "Canonical agree", "Unknown"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor(colors[r - 1])
    _save(fig, charts / "paired_parity_subset.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, help="Directory with requests.csv and provider_summary.json")
    parser.add_argument("--charts", required=True, help="Output chart directory")
    args = parser.parse_args()

    analysis = Path(args.analysis)
    charts = Path(args.charts)
    charts.mkdir(parents=True, exist_ok=True)
    for old_chart in charts.glob("*.png"):
        old_chart.unlink()

    rows = _load_rows(analysis / "requests.csv")
    per_item_rows = _load_rows(analysis / "per_item.csv")
    summary = json.loads((analysis / "provider_summary.json").read_text(encoding="utf-8"))

    provider_scorecard_table(summary, charts)
    request_outcomes(summary, charts)
    throttling_diagnostics(summary, charts)
    latency_boxplot(rows, summary, charts)
    ttft_p50_p95(summary, charts)
    output_tokens_boxplot(rows, summary, charts)
    parity_table(per_item_rows, charts)

    created = sorted(str(p) for p in charts.glob("*.png"))
    print(json.dumps({"charts": created, "n_charts": len(created)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())