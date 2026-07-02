"""Normalize GLMsBench JSON output into CSV and summary artifacts.

Usage:
    python3 scripts/normalize_results.py \
      --results results/arc-n30/raw/results.json \
      --out results/arc-n30/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glmsbench.scoring.exact import extract_letter
from glmsbench.scoring.numeric import extract_number


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
    }


def _extract_answer(suite: str, output: str) -> str | None:
    if suite in {"arc", "mmlu"}:
        return extract_letter(output)
    if suite == "gsm8k":
        n = extract_number(output)
        return str(n) if n is not None else None
    return None


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    timing = record.get("timing") or {}
    rate = record.get("rate") or {}
    output = record.get("output") or ""
    prompt = record.get("prompt") or ""
    input_tokens = timing.get("input_tokens") or 0
    output_tokens = timing.get("output_tokens") or 0
    return {
        "provider": record.get("provider"),
        "suite": record.get("suite"),
        "item_id": record.get("item_id"),
        "pass_type": record.get("pass_type"),
        "pass_index": record.get("pass_index"),
        "ok": record.get("error") is None and record.get("http_status") == 200 and bool(timing),
        "http_status": record.get("http_status"),
        "error": record.get("error"),
        "prompt_chars": len(prompt),
        "output_chars": len(output),
        "extracted_answer": _extract_answer(record.get("suite", ""), output),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": timing.get("reasoning_tokens") or 0,
        "total_tokens": input_tokens + output_tokens,
        "ttft_ms": timing.get("ttft_ms"),
        "inter_token_latency_ms": timing.get("inter_token_latency_ms"),
        "e2e_ms": timing.get("e2e_ms"),
        "stop_reason": timing.get("stop_reason"),
        "n_attempts": rate.get("n_attempts") or 0,
        "n_throttled": rate.get("n_throttled") or 0,
        "total_backoff_ms": rate.get("total_backoff_ms") or 0.0,
        "had_backoff": bool(rate.get("had_backoff")),
        "raw_output_preview": output[:300].replace("\n", " "),
    }


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _num(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _provider_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for provider in sorted({str(r["provider"]) for r in rows}):
        rs = [r for r in rows if r["provider"] == provider]
        ok = [r for r in rs if r["ok"]]
        out[provider] = {
            "requests": len(rs),
            "ok_requests": len(ok),
            "error_requests": len(rs) - len(ok),
            "input_tokens": _stats(_values(ok, "input_tokens")),
            "output_tokens": _stats(_values(ok, "output_tokens")),
            "total_tokens": _stats(_values(ok, "total_tokens")),
            "ttft_ms": _stats(_values(ok, "ttft_ms")),
            "e2e_ms": _stats(_values(ok, "e2e_ms")),
            "inter_token_latency_ms": _stats(_values(ok, "inter_token_latency_ms")),
            "throttled_events": sum(int(r["n_throttled"] or 0) for r in rs),
            "backoff_ms": sum(float(r["total_backoff_ms"] or 0.0) for r in rs),
            "avg_attempts": statistics.mean([float(r["n_attempts"] or 0) for r in rs]) if rs else 0,
        }
    return out


def _parity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    providers = sorted({str(r["provider"]) for r in rows})
    suites = sorted({str(r["suite"]) for r in rows})
    out: dict[str, Any] = {}
    if len(providers) < 2:
        return out
    a, b = providers[:2]
    for suite in suites:
        parity_rows = [r for r in rows if r["suite"] == suite and r["pass_type"] == "parity" and r["ok"]]
        a_items = {r["item_id"]: r for r in parity_rows if r["provider"] == a}
        b_items = {r["item_id"]: r for r in parity_rows if r["provider"] == b}
        shared = sorted(set(a_items) & set(b_items))
        pairs = []
        canonical_disagree = 0
        unknown = 0
        raw_agree = 0
        for item_id in shared:
            ar = a_items[item_id]
            br = b_items[item_id]
            if ar["raw_output_preview"] == br["raw_output_preview"]:
                raw_agree += 1
            aa = ar.get("extracted_answer")
            ba = br.get("extracted_answer")
            if aa is None or ba is None:
                unknown += 1
            elif aa != ba:
                canonical_disagree += 1
            pairs.append({
                "suite": suite,
                "item_id": item_id,
                f"{a}_answer": aa,
                f"{b}_answer": ba,
                "canonical_agree": (aa is not None and ba is not None and aa == ba),
                "canonical_unknown": aa is None or ba is None,
            })
        n = len(shared)
        out[suite] = {
            "providers": [a, b],
            "n_compared": n,
            "canonical_disagreements": canonical_disagree,
            "canonical_unknown": unknown,
            "canonical_disagreement_rate": canonical_disagree / n if n else None,
            "raw_text_agreements": raw_agree,
            "raw_text_agreement_rate": raw_agree / n if n else None,
            "pairs": pairs,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to GLMsBench results.json")
    parser.add_argument("--out", required=True, help="Analysis output directory")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [_flatten_record(r) for r in report.get("requests", [])]
    if not rows:
        raise SystemExit(f"No request records found in {results_path}")

    requests_csv = out_dir / "requests.csv"
    with requests_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    parity = _parity_summary(rows)
    per_item_rows = []
    for suite_data in parity.values():
        per_item_rows.extend(suite_data.get("pairs", []))
    if per_item_rows:
        fieldnames = sorted({k for row in per_item_rows for k in row})
        with (out_dir / "per_item.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_item_rows)

    summary = {
        "source_results": str(results_path),
        "n_requests": len(rows),
        "providers": sorted({r["provider"] for r in rows}),
        "suites": sorted({r["suite"] for r in rows}),
        "provider_summary": _provider_summary(rows),
        "parity_summary": parity,
        "upstream_aggregates": report.get("aggregates", {}),
    }
    (out_dir / "provider_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "requests_csv": str(requests_csv),
        "provider_summary": str(out_dir / "provider_summary.json"),
        "per_item_csv": str(out_dir / "per_item.csv") if per_item_rows else None,
        "n_requests": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
