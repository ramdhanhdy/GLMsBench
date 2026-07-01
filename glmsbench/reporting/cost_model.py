from __future__ import annotations
from typing import TYPE_CHECKING
from ..metrics import effective_cost_per_million

if TYPE_CHECKING:
    from ..config import Pricing, UsageTier


def build_cost_comparison(
    providers_pricing: dict[str, "Pricing"],
    usage_tiers: dict[str, "UsageTier"],
) -> dict:
    """Cross-provider cost comparison, machine-readable.

    Metered providers report their flat $/1M input/output rates as-is.
    Subscription providers report their flat monthly fee plus an effective
    $/1M-token range per usage tier (see metrics.effective_cost_per_million).
    """
    out: dict = {
        "usage_tiers": {
            name: {"tokens_per_day_min": t.tokens_per_day_min, "tokens_per_day_max": t.tokens_per_day_max}
            for name, t in usage_tiers.items()
        },
        "providers": {},
    }
    for name, pricing in providers_pricing.items():
        if pricing.mode == "metered":
            out["providers"][name] = {
                "mode": "metered",
                "input_usd_per_million": pricing.input,
                "output_usd_per_million": pricing.output,
            }
        else:
            by_tier = {}
            for tier_name, tier in usage_tiers.items():
                eff = effective_cost_per_million(
                    pricing.monthly_usd, tier.tokens_per_day_min, tier.tokens_per_day_max
                )
                by_tier[tier_name] = eff
            out["providers"][name] = {
                "mode": "subscription",
                "monthly_usd": pricing.monthly_usd,
                "effective_usd_per_million_by_tier": by_tier,
            }
    return out
