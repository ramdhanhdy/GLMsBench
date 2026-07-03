"""Analyze labeled benchmark quality for a GLMsBench run.

Usage:
    python3 scripts/analyze_quality.py \
      --run-dir results/arc-n30-merged-zai-umans-c2-20260702T112934Z

For ARC, this joins request records to ARC-Challenge gold labels and reports:
- strict task success: parseable correct answers / all requests
- answered-only accuracy: parseable correct answers / parseable answers
- empty/capped reasoning failures
- pass-to-pass stability by item and provider
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glmsbench.datasets.arc import ARCLoader
from glmsbench.scoring.exact import extract_letter


def _pct(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def _ok(record: dict[str, Any]) -> bool:
    return record.get("error") is None and record.get("http_status") == 200 and bool(record.get("timing"))


def _timing(record: dict[str, Any], key: str) -> float | None:
    timing = record.get("timing") or {}
    value = timing.get(key)
    return float(value) if value is not None else None


def _short_question(prompt: str) -> str:
    body = prompt.split("\n\n", 1)[-1]
    return body.split("\nA.", 1)[0].replace("Answer:", "").strip()


def _load_gold_for_arc(n_items: int) -> dict[str, Any]:
    items = ARCLoader().load(n_items)
    return {item.id: item for item in items}


def _infer_arc_n(records: list[dict[str, Any]]) -> int:
    max_index = -1
    for record in records:
        item_id = str(record.get("item_id", ""))
        match = re.match(r"arc-(\d+)$", item_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    if max_index < 0:
        raise SystemExit("Cannot infer ARC item count from item_id values")
    return max_index + 1


def _completion_budget_from_records(records: list[dict[str, Any]]) -> int | None:
    values: list[float] = []
    for record in records:
        value = (record.get("timing") or {}).get("output_tokens")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return int(max(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_quality(run_dir: Path) -> dict[str, Any]:
    raw_path = run_dir / "raw" / "results.json"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}")

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    records = data.get("requests", [])
    if not records:
        raise SystemExit(f"No request records found in {raw_path}")

    suites = sorted({str(r.get("suite")) for r in records})
    if suites != ["arc"]:
        raise SystemExit(f"Quality analysis currently supports only ARC runs, got: {suites}")

    gold_items = _load_gold_for_arc(_infer_arc_n(records))
    gold = {item_id: item.gold for item_id, item in gold_items.items()}
    providers = sorted({str(r.get("provider")) for r in records})

    enriched: list[dict[str, Any]] = []
    for record in records:
        item_id = str(record.get("item_id"))
        parsed = extract_letter(record.get("output") or "")
        item_gold = gold.get(item_id)
        enriched.append({
            **record,
            "_ok": _ok(record),
            "_answer": parsed,
            "_gold": item_gold,
            "_correct": parsed == item_gold if parsed is not None and item_gold else None,
            "_exact_single_letter": (record.get("output") or "").strip() in {"A", "B", "C", "D"},
        })

    item_rows: list[dict[str, Any]] = []
    for item_id in sorted(gold):
        prompt = gold_items[item_id].prompt
        row: dict[str, Any] = {
            "item_id": item_id,
            "gold": gold[item_id],
            "question": _short_question(prompt),
        }
        for provider in providers:
            prs = [r for r in enriched if r["item_id"] == item_id and r["provider"] == provider]
            answers = [r["_answer"] for r in prs]
            corrects = [r["_correct"] for r in prs]
            row[f"{provider}_answers"] = "|".join(str(a) if a is not None else "NULL" for a in answers)
            row[f"{provider}_correct"] = sum(1 for c in corrects if c is True)
            row[f"{provider}_unknown"] = sum(1 for a in answers if a is None)
            row[f"{provider}_stable"] = len(set(a for a in answers if a is not None)) <= 1 and all(a is not None for a in answers)
        item_rows.append(row)

    summary: dict[str, Any] = {
        "source_results": str(raw_path),
        "suites": suites,
        "providers": providers,
        "n_requests": len(enriched),
        "n_items": len(gold),
        "completion_budget_tokens": _completion_budget_from_records(enriched),
        "provider_quality": {},
        "item_summary": {},
        "gold_distribution": dict(sorted(Counter(gold.values()).items())),
        "opportunities": [
            "Add gold labels to normalization output so accuracy/correctness is first-class, not a separate join.",
            "Report request-level accuracy, item-level accuracy, answered-only accuracy, and unknown parse rate separately.",
            "Track pass-to-pass stability: parity and latency passes can disagree even at temperature 0.",
            "Separate benchmark capability from provider reliability: correctness only among completed requests hides failed-request impact.",
            "Analyze hard items: list items missed by both providers, missed by only one provider, and unknown/non-letter outputs.",
            "Measure latency-quality coupling: compare latency/token counts for correct vs incorrect/unknown outputs.",
            "Scale beyond the current sample when tighter confidence intervals are needed.",
            "Store dataset metadata (question, choices, gold, category) in raw results for reproducible audits without reloading HF.",
        ],
    }

    for provider in providers:
        prs = [r for r in enriched if r["provider"] == provider]
        parseable = [r for r in prs if r["_answer"] is not None]
        correct = [r for r in parseable if r["_correct"] is True]
        incorrect = [r for r in parseable if r["_correct"] is False]
        unknown = [r for r in prs if r["_answer"] is None]
        exact_single = [r for r in prs if r["_exact_single_letter"]]
        length_stops = [r for r in prs if (r.get("timing") or {}).get("stop_reason") == "length"]
        empty_content = [r for r in prs if not (r.get("output") or "")]
        non_empty = [r for r in prs if r.get("output")]

        item_all_correct = 0
        item_stable_correct = 0
        item_both_wrong_same = 0
        unstable_or_unknown: list[str] = []
        wrong_stable: list[str] = []
        for item_id in sorted(gold):
            rs = [r for r in prs if r["item_id"] == item_id]
            answers = [r["_answer"] for r in rs]
            corrects = [r["_correct"] for r in rs]
            if rs and all(c is True for c in corrects):
                item_all_correct += 1
            if len(set(answers)) == 1 and answers[0] is not None:
                if answers[0] == gold[item_id]:
                    item_stable_correct += 1
                else:
                    item_both_wrong_same += 1
                    wrong_stable.append(item_id)
            else:
                unstable_or_unknown.append(item_id)

        empty_e2e = [_timing(r, "e2e_ms") for r in empty_content]
        non_empty_e2e = [_timing(r, "e2e_ms") for r in non_empty]
        correct_e2e = [_timing(r, "e2e_ms") for r in correct]
        unknown_e2e = [_timing(r, "e2e_ms") for r in unknown]

        summary["provider_quality"][provider] = {
            "requests": len(prs),
            "ok_requests": sum(1 for r in prs if r["_ok"]),
            "parseable_answers": len(parseable),
            "unknown_answers": len(unknown),
            "empty_content": len(empty_content),
            "correct_requests": len(correct),
            "incorrect_requests": len(incorrect),
            "strict_task_success": _pct(len(correct), len(prs)),
            "answered_only_accuracy": _pct(len(correct), len(parseable)),
            "exact_single_letter_outputs": len(exact_single),
            "length_stops": len(length_stops),
            "items_all_correct": item_all_correct,
            "items_stable_correct": item_stable_correct,
            "items_both_wrong_same": item_both_wrong_same,
            "wrong_stable_items": wrong_stable,
            "unstable_or_unknown_items": unstable_or_unknown,
            "e2e_ms_non_empty_median": _median([x for x in non_empty_e2e if x is not None]),
            "e2e_ms_empty_median": _median([x for x in empty_e2e if x is not None]),
            "e2e_ms_correct_median": _median([x for x in correct_e2e if x is not None]),
            "e2e_ms_unknown_median": _median([x for x in unknown_e2e if x is not None]),
            "answer_distribution": dict(sorted(Counter(r["_answer"] if r["_answer"] is not None else "UNKNOWN" for r in prs).items())),
        }

    classes: dict[str, list[str]] = {}
    for row in item_rows:
        status: list[str] = []
        for provider in providers:
            correct_count = int(row[f"{provider}_correct"])
            unknown_count = int(row[f"{provider}_unknown"])
            if correct_count == 2:
                status.append(f"{provider}:2/2 correct")
            elif correct_count == 1 and unknown_count == 1:
                status.append(f"{provider}:1 correct + 1 empty")
            elif correct_count == 0 and unknown_count == 2:
                status.append(f"{provider}:2 empty")
            else:
                status.append(f"{provider}:{correct_count} correct, {unknown_count} empty")
        if all("2/2 correct" in s for s in status):
            key = "Both providers 2/2 correct"
        elif any("2 empty" in s for s in status) and any("2/2 correct" in s for s in status):
            key = "One provider fully correct, other fully empty"
        elif any("1 correct + 1 empty" in s for s in status):
            key = "At least one provider unstable: one pass correct, one pass empty"
        elif any("2 empty" in s for s in status):
            key = "Both providers have full-empty item or severe cap issue"
        else:
            key = "Other mixed outcome"
        classes.setdefault(key, []).append(row["item_id"])

    both_all_correct = classes.get("Both providers 2/2 correct", [])
    items_with_unknown = [
        row["item_id"]
        for row in item_rows
        if any(int(row[f"{provider}_unknown"]) > 0 for provider in providers)
    ]
    summary["item_summary"] = {
        "both_providers_all_passes_correct": len(both_all_correct),
        "items_with_any_unknown_answer": items_with_unknown,
        "outcome_classes": classes,
    }

    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "item_correctness.csv", item_rows)
    _write_quality_reports(out_dir, summary, item_rows, enriched)
    return summary


def _write_quality_reports(out_dir: Path, summary: dict[str, Any], item_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    providers = summary["providers"]
    quality = summary["provider_quality"]

    budget = summary.get("completion_budget_tokens")
    budget_text = f"{budget}-token completion budget" if budget else "completion budget"

    lines: list[str] = []
    lines.append(f"# GLMsBench ARC n={summary['n_items']} Quality Analysis")
    lines.append("")
    lines.append("This analysis joins provider outputs to ARC-Challenge gold labels through the existing `ARCLoader`. It covers correctness, parseability, item-level stability, and quality/latency coupling.")
    lines.append("")
    lines.append("## Provider accuracy")
    lines.append("")
    lines.append("| Provider | Correct / requests | Strict success | Answered-only accuracy | Empty content | Exact single-letter | Length stops | Stable correct items |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for provider in providers:
        row = quality[provider]
        lines.append(
            f"| {provider} | {row['correct_requests']} / {row['requests']} | {_fmt_pct(row['strict_task_success'])} | "
            f"{_fmt_pct(row['answered_only_accuracy'])} | {row['empty_content']} | {row['exact_single_letter_outputs']} | "
            f"{row['length_stops']} | {row['items_stable_correct']} / {summary['n_items']} |"
        )
    lines.append("")
    lines.append("## Provider latency by quality bucket")
    lines.append("")
    lines.append("| Provider | Median E2E non-empty | Median E2E empty | Median E2E correct | Median E2E unknown |")
    lines.append("|---|---:|---:|---:|---:|")
    for provider in providers:
        row = quality[provider]
        lines.append(
            f"| {provider} | {_fmt_ms(row['e2e_ms_non_empty_median'])} | {_fmt_ms(row['e2e_ms_empty_median'])} | "
            f"{_fmt_ms(row['e2e_ms_correct_median'])} | {_fmt_ms(row['e2e_ms_unknown_median'])} |"
        )
    lines.append("")
    lines.append("## Item patterns")
    lines.append("")
    lines.append(f"- Items where both providers were correct on both passes: **{summary['item_summary']['both_providers_all_passes_correct']}/{summary['n_items']}**")
    unknown_items = summary["item_summary"]["items_with_any_unknown_answer"]
    lines.append(f"- Items with any empty/non-parseable output: **{len(unknown_items)}/{summary['n_items']}** - {', '.join(unknown_items) if unknown_items else 'none'}")
    lines.append("")
    lines.append("## Gold answer distribution")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|---|---:|")
    for label, count in summary["gold_distribution"].items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## Analysis gaps unlocked by current data")
    lines.append("")
    for item in summary["opportunities"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `quality/quality_summary.json`")
    lines.append("- `quality/item_correctness.csv`")
    lines.append("- `quality/deep_quality_report.md`")
    (out_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    deep: list[str] = []
    deep.append("# Additional Quality Analysis: ARC Correctness and Failure Modes")
    deep.append("")
    deep.append(f"This is a deeper read of the existing merged ARC n={summary['n_items']} data. It joins raw provider outputs with ARC gold labels and separates answer correctness from operational / generation failures.")
    deep.append("")
    deep.append("## Headline")
    deep.append("")
    deep.append(f"- Among responses that produced a parseable answer, provider answered-only accuracy stayed high on this ARC n={summary['n_items']} sample, but parsed wrong answers still occurred.")
    deep.append(f"- Empty final answers after spending the whole {budget_text} on reasoning are a separate failure mode from wrong parsed answers.")
    deep.append(f"- Therefore the current benchmark is measuring a mix of: (1) ARC capability, (2) answer-format reliability, and (3) whether the {budget_text} is enough when reasoning is enabled.")
    deep.append("- Exact single-letter compliance varies by provider; see the provider-level table and empty-content table below for request-level details.")
    deep.append("")
    deep.append("## Provider-level correctness")
    deep.append("")
    deep.append("| Provider | Requests | Parseable correct | Empty content | Strict success rate | Answered-only accuracy | Exact single-letter outputs | Length stops | Median E2E non-empty | Median E2E empty |")
    deep.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for provider in providers:
        row = quality[provider]
        deep.append(
            f"| {provider} | {row['requests']} | {row['correct_requests']} | {row['empty_content']} | "
            f"{_fmt_pct(row['strict_task_success'])} | {_fmt_pct(row['answered_only_accuracy'])} | "
            f"{row['exact_single_letter_outputs']} | {row['length_stops']} | "
            f"{_fmt_ms(row['e2e_ms_non_empty_median'])} | {_fmt_ms(row['e2e_ms_empty_median'])} |"
        )
    deep.append("")
    deep.append("## Empty-content failures")
    deep.append("")
    deep.append("| Provider | Item | Pass | Gold | Stop | Output tokens | Reasoning tokens | E2E | Question short |")
    deep.append("|---|---|---|---|---|---:|---:|---:|---|")
    item_by_id = {row["item_id"]: row for row in item_rows}
    for record in records:
        if record.get("output"):
            continue
        timing = record.get("timing") or {}
        item_row = item_by_id[record["item_id"]]
        question = item_row["question"][:95].replace("|", "/").replace("\n", " ")
        deep.append(
            f"| {record['provider']} | {record['item_id']} | {record['pass_type']} | {item_row['gold']} | "
            f"{timing.get('stop_reason')} | {timing.get('output_tokens')} | {timing.get('reasoning_tokens')} | "
            f"{(timing.get('e2e_ms') or 0):,.0f} ms | {question}... |"
        )
    deep.append("")
    deep.append("## Item-level outcome taxonomy")
    deep.append("")
    deep.append("| Outcome | Count | Items |")
    deep.append("|---|---:|---|")
    for label, items in summary["item_summary"]["outcome_classes"].items():
        deep.append(f"| {label} | {len(items)} | {', '.join(items)} |")
    deep.append("")
    deep.append("## What else we can analyze with existing data")
    deep.append("")
    for item in [
        "Strict task success: answer emitted and correct, not just HTTP 200.",
        "Format adherence / answerability: empty output, non-letter output, multiple-letter output, and exact single-letter compliance.",
        "Reasoning budget failures: stop_reason=length, output_tokens=max_tokens, reasoning_tokens≈max_tokens.",
        "Pass-to-pass stability at temperature 0: parity vs latency pass per item/provider.",
        "Latency-quality coupling: empty/capped outputs are much slower, especially on umans.",
        "Provider-specific hard items: items where one provider emits an answer and the other exhausts reasoning budget.",
        "Prompt/item sensitivity: correlate failure with question length, answer label, and scientific subtopic (needs metadata/classification).",
        "Benchmark design sensitivity: rerun only failed/empty items with larger max_tokens or reasoning_effort=none to separate model knowledge from harness settings.",
        "Confidence in conclusions: bootstrap CIs and scale further if tighter intervals are needed.",
        "Data schema improvement: store gold labels, choices, and scoring fields inside results.json so quality analysis is reproducible without reloading HF.",
    ]:
        deep.append(f"- {item}")
    deep.append("")
    (out_dir / "deep_quality_report.md").write_text("\n".join(deep) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory containing raw/results.json")
    args = parser.parse_args()
    summary = analyze_quality(Path(args.run_dir))
    out_dir = Path(args.run_dir) / "quality"
    print(json.dumps({
        "quality_summary": str(out_dir / "quality_summary.json"),
        "item_correctness": str(out_dir / "item_correctness.csv"),
        "quality_report": str(out_dir / "quality_report.md"),
        "deep_quality_report": str(out_dir / "deep_quality_report.md"),
        "providers": summary["providers"],
        "n_requests": summary["n_requests"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
