from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING
from ..models import RequestRecord
from ..metrics import latency_stats, cost_total, effective_cost_per_million
from ..scoring.parity import compare_text_parity
from ..scoring.exact import extract_letter
from ..scoring.numeric import extract_number

if TYPE_CHECKING:
    from ..config import Pricing, UsageTier


def _extractor_for(suite: str) -> Optional[Callable[[str], Optional[str]]]:
    """Pick the canonical-answer extractor for a suite.

    MCQ suites (mmlu, arc) -> letter; numeric suite (gsm8k) -> number.
    humaneval has no text answer key, so parity there is pass/fail based
    (handled separately); text parity returns None.
    """
    if suite in ("mmlu", "arc"):
        return extract_letter
    if suite == "gsm8k":
        return lambda text: (str(extract_number(text)) if extract_number(text) is not None else None)
    return None


def build_markdown(
    records: list[RequestRecord],
    providers_pricing: Optional[dict[str, "Pricing"]] = None,
    usage_tiers: Optional[dict[str, "UsageTier"]] = None,
) -> str:
    providers = sorted({r.provider for r in records})
    suites = sorted({r.suite for r in records})
    lines: list[str] = ["# GLM-5.2 Provider Benchmark\n"]

    # --- Accuracy & Parity (computed from parity passes) ---
    lines.append("## Accuracy & Parity\n")
    lines.append(
        "| Suite | " + " | ".join(f"{p} (n)" for p in providers)
        + " | Disagreement (canonical) | Raw-text agreement |"
    )
    lines.append("|" + "---|" * (len(providers) + 2))
    for suite in suites:
        row = [suite]
        for prov in providers:
            parity = [r for r in records if r.provider == prov and r.suite == suite
                      and r.pass_type == "parity" and r.ok]
            row.append(str(len(parity)))
        canon_dis, raw_agree = _parity_rates(records, suite, providers)
        row.append(f"{canon_dis * 100:.1f}%" if canon_dis is not None else "-")
        row.append(f"{raw_agree * 100:.1f}%" if raw_agree is not None else "-")
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
        pricing = (providers_pricing or {}).get(prov)
        if pricing and pricing.mode == "subscription":
            lines.append(f"| {prov} | flat ${pricing.monthly_usd:.2f}/mo (subscription, see tier table) |")
        else:
            lines.append(f"| {prov} | {cost_total(records, prov):.4f} |")
    lines.append("")

    # --- Effective Cost by Usage Tier (subscription providers only) ---
    subscription_provs = [p for p in providers
                           if (providers_pricing or {}).get(p) and providers_pricing[p].mode == "subscription"]
    if subscription_provs and usage_tiers:
        lines.append("## Effective Cost by Usage Tier\n")
        lines.append("| Provider | Plan | Usage Tier | $/1M tokens (high usage \u2013 low usage) |")
        lines.append("|---|---|---|---|")
        for prov in subscription_provs:
            pricing = providers_pricing[prov]
            for tier_name, tier in usage_tiers.items():
                eff = effective_cost_per_million(
                    pricing.monthly_usd, tier.tokens_per_day_min, tier.tokens_per_day_max
                )
                if eff is None:
                    continue
                lines.append(
                    f"| {prov} | ${pricing.monthly_usd:.0f}/mo | {tier_name} | "
                    f"${eff['high_usage_usd_per_million']:.4f} \u2013 ${eff['low_usage_usd_per_million']:.4f} |"
                )
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


def _parity_rates(records, suite, providers):
    """Return (canonical disagreement rate, raw-text agreement rate).

    For text-answer suites (mmlu/arc/gsm8k), canonical disagreement compares the
    extracted answer (the spec §6.2 mandated comparison). For humaneval there is
    no text key, so canonical returns None; raw-text agreement is still reported
    as a coarse signal.
    """
    if len(providers) < 2:
        return None, None
    a, b = providers[0], providers[1]
    a_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == a and r.suite == suite and r.pass_type == "parity" and r.ok}
    b_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == b and r.suite == suite and r.pass_type == "parity" and r.ok}
    shared = set(a_items) & set(b_items)
    if not shared:
        return None, None

    raw_agree = sum(1 for k in shared if a_items[k] == b_items[k]) / len(shared)

    extract_fn = _extractor_for(suite)
    if extract_fn is None:
        # humaneval: no canonical text answer -> only raw-text (coarse) is reportable.
        return None, raw_agree

    disagree = 0
    for k in shared:
        res = compare_text_parity(None, a_items[k], b_items[k], extract_fn)
        if not res["answer_agree"]:
            disagree += 1
    return disagree / len(shared), raw_agree


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"
