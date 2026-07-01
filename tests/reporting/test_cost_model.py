from glmsbench.config import Pricing, UsageTier
from glmsbench.reporting.cost_model import build_cost_comparison


def test_metered_provider_reports_flat_rates():
    pricing = {"zai": Pricing(mode="metered", input=1.4, output=4.4)}
    out = build_cost_comparison(pricing, {})
    assert out["providers"]["zai"]["mode"] == "metered"
    assert out["providers"]["zai"]["input_usd_per_million"] == 1.4
    assert out["providers"]["zai"]["output_usd_per_million"] == 4.4


def test_subscription_provider_reports_effective_range_per_tier():
    pricing = {"umans": Pricing(mode="subscription", monthly_usd=50.0)}
    tiers = {
        "early_setup": UsageTier(tokens_per_day_min=50_000_000, tokens_per_day_max=90_000_000),
        "maximized": UsageTier(tokens_per_day_min=100_000_000, tokens_per_day_max=300_000_000),
    }
    out = build_cost_comparison(pricing, tiers)
    entry = out["providers"]["umans"]
    assert entry["mode"] == "subscription"
    assert entry["monthly_usd"] == 50.0
    assert set(entry["effective_usd_per_million_by_tier"]) == {"early_setup", "maximized"}
    # maximized (more tokens/day) should be cheaper per-million than early_setup
    maximized = entry["effective_usd_per_million_by_tier"]["maximized"]
    early = entry["effective_usd_per_million_by_tier"]["early_setup"]
    assert maximized["high_usage_usd_per_million"] < early["low_usage_usd_per_million"]
    assert out["usage_tiers"]["maximized"]["tokens_per_day_max"] == 300_000_000


def test_mixed_providers_both_reported():
    pricing = {
        "zai": Pricing(mode="metered", input=1.4, output=4.4),
        "umans": Pricing(mode="subscription", monthly_usd=50.0),
    }
    tiers = {"maximized": UsageTier(tokens_per_day_min=100_000_000, tokens_per_day_max=300_000_000)}
    out = build_cost_comparison(pricing, tiers)
    assert out["providers"]["zai"]["mode"] == "metered"
    assert out["providers"]["umans"]["mode"] == "subscription"
