from __future__ import annotations
from ..models import RequestRecord
from ..metrics import latency_stats, cost_total


def build_markdown(records: list[RequestRecord]) -> str:
    providers = sorted({r.provider for r in records})
    suites = sorted({r.suite for r in records})
    lines: list[str] = ["# GLM-5.2 Provider Benchmark\n"]

    # --- Accuracy & Parity (computed from parity passes) ---
    lines.append("## Accuracy & Parity\n")
    lines.append("| Suite | " + " | ".join(f"{p} (n)" for p in providers) + " | Disagreement |")
    lines.append("|" + "---|" * (len(providers) + 1))
    for suite in suites:
        row = [suite]
        for prov in providers:
            parity = [r for r in records if r.provider == prov and r.suite == suite
                      and r.pass_type == "parity" and r.ok]
            row.append(str(len(parity)))
        dis = _disagreement_rate(records, suite, providers)
        row.append(f"{dis * 100:.1f}%" if dis is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Latency ---
    lines.append("## Latency\n")
    lines.append("| Suite | Provider | Scope | TTFT p50 (ms) | E2E p50 (ms) | n |")
    lines.append("|---|---|---|---|---|---|")
    for suite in suites:
        for prov in providers:
            stats = latency_stats(records, prov, suite)
            for scope in ("clean", "effective"):
                s = stats[scope]
                ttft = s["ttft_ms"].get("median") if isinstance(s.get("ttft_ms"), dict) else None
                e2e = s["e2e_ms"].get("median") if isinstance(s.get("e2e_ms"), dict) else None
                lines.append(f"| {suite} | {prov} | {scope} | "
                             f"{_fmt(ttft)} | {_fmt(e2e)} | {s.get('n', 0)} |")
    lines.append("")

    # --- Cost ---
    lines.append("## Cost\n")
    lines.append("| Provider | Total USD |")
    lines.append("|---|---|")
    for prov in providers:
        lines.append(f"| {prov} | {cost_total(records, prov):.4f} |")
    lines.append("")

    # --- Rate Limits ---
    lines.append("## Rate Limits\n")
    lines.append("| Provider | Throttled (429) | Avg retries | Backoff (ms) |")
    lines.append("|---|---|---|---|")
    for prov in providers:
        recs = [r for r in records if r.provider == prov and r.rate]
        throttled = sum(r.rate.n_throttled for r in recs)
        backoff = sum(r.rate.total_backoff_ms for r in recs)
        avg_retries = (sum(r.rate.n_attempts - 1 for r in recs) / len(recs)) if recs else 0
        lines.append(f"| {prov} | {throttled} | {avg_retries:.2f} | {backoff:.0f} |")
    lines.append("")

    return "\n".join(lines)


def _disagreement_rate(records, suite, providers):
    """Fraction of parity items where the two providers' outputs differ."""
    if len(providers) < 2:
        return None
    a, b = providers[0], providers[1]
    a_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == a and r.suite == suite and r.pass_type == "parity" and r.ok}
    b_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == b and r.suite == suite and r.pass_type == "parity" and r.ok}
    shared = set(a_items) & set(b_items)
    if not shared:
        return None
    disagree = sum(1 for k in shared if a_items[k] != b_items[k])
    return disagree / len(shared)


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"
