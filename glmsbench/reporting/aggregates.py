from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from ..models import RequestRecord
from ..metrics import latency_stats, cost_total
from ..scoring.parity import compare_text_parity
from ..scoring.exact import extract_letter
from ..scoring.numeric import extract_number
from .cost_model import build_cost_comparison

if TYPE_CHECKING:
    from ..config import Pricing, UsageTier


def _extractor_for(suite: str):
    if suite in ("mmlu", "arc"):
        return extract_letter
    if suite == "gsm8k":
        return lambda text: (str(extract_number(text)) if extract_number(text) is not None else None)
    return None


def build_aggregates(
    records: list[RequestRecord],
    providers_pricing: Optional[dict[str, "Pricing"]] = None,
    usage_tiers: Optional[dict[str, "UsageTier"]] = None,
) -> dict:
    """Compute per-(provider,suite) latency/cost and cross-provider parity.

    Mirrors the Markdown report's logic but in machine-readable form so the JSON
    report carries precomputed aggregates (spec §9), not just raw records.

    `providers_pricing`/`usage_tiers` are optional; when provided, a
    `cost_by_tier` block is added covering subscription-priced providers'
    effective $/1M-token range under each usage tier (metered providers are
    already covered by the additive `cost` block).
    """
    providers = sorted({r.provider for r in records})
    suites = sorted({r.suite for r in records})

    aggregates: dict = {"by_suite": {}, "cost": {}}

    for suite in suites:
        aggregates["by_suite"][suite] = {}
        for prov in providers:
            stats = latency_stats(records, prov, suite)
            aggregates["by_suite"][suite][prov] = {
                "latency": {
                    "clean": stats["clean"],
                    "effective": stats["effective"],
                },
            }
        # parity across the two providers
        if len(providers) >= 2:
            aggregates["by_suite"][suite]["parity"] = _parity(records, suite, providers)

    for prov in providers:
        aggregates["cost"][prov] = cost_total(records, prov)

    if providers_pricing:
        aggregates["cost_by_tier"] = build_cost_comparison(providers_pricing, usage_tiers or {})

    return aggregates


def _parity(records, suite, providers):
    a, b = providers[0], providers[1]
    a_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == a and r.suite == suite and r.pass_type == "parity" and r.ok}
    b_items = {(r.suite, r.item_id): r.output for r in records
               if r.provider == b and r.suite == suite and r.pass_type == "parity" and r.ok}
    shared = set(a_items) & set(b_items)
    out: dict = {"n_compared": len(shared)}
    if not shared:
        out["canonical_disagreement_rate"] = None
        out["raw_text_agreement_rate"] = None
        return out

    raw_agree = sum(1 for k in shared if a_items[k] == b_items[k]) / len(shared)
    out["raw_text_agreement_rate"] = raw_agree

    extract_fn = _extractor_for(suite)
    if extract_fn is None:
        out["canonical_disagreement_rate"] = None
        return out

    disagree = 0
    for k in shared:
        res = compare_text_parity(None, a_items[k], b_items[k], extract_fn)
        if not res["answer_agree"]:
            disagree += 1
    out["canonical_disagreement_rate"] = disagree / len(shared)
    return out
