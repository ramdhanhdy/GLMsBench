from glmsbench.config import Pricing, UsageTier
from glmsbench.models import RequestRecord, Timing, RateStats
from glmsbench.reporting.markdown import build_markdown


def _rec(provider, suite, item_id, pass_type="parity", output="A",
         ttft=100, e2e=200, out_tok=1, in_tok=5, had_backoff=False, cost=0.01):
    return RequestRecord(
        provider=provider, suite=suite, item_id=item_id, pass_type=pass_type, pass_index=0,
        prompt="p", output=output,
        timing=Timing(ttft_ms=ttft, e2e_ms=e2e, output_tokens=out_tok, input_tokens=in_tok),
        rate=RateStats(had_backoff=had_backoff), cost_usd=cost, http_status=200, error=None,
    )


def test_markdown_has_required_sections():
    recs = [
        _rec("zai", "mmlu", "mmlu-0000", output="A"),
        _rec("umans", "mmlu", "mmlu-0000", output="A"),
    ]
    md = build_markdown(recs)
    assert "# GLM-5.2 Provider Benchmark" in md
    assert "Accuracy" in md
    assert "Parity" in md
    assert "Latency" in md
    assert "Cost" in md
    assert "Rate Limits" in md


def test_markdown_reports_disagreement():
    recs = [
        _rec("zai", "mmlu", "mmlu-0000", output="A"),
        _rec("umans", "mmlu", "mmlu-0000", output="C"),
    ]
    md = build_markdown(recs)
    assert "100.0%" in md or "100%" in md  # 1/1 disagreement


def test_canonical_parity_not_inflated_by_prose_differences():
    """Regression guard for the §6.2 bug: two providers that emit different
    prose but the SAME canonical answer must NOT be counted as a disagreement.
    Raw text differs, canonical agrees -> canonical disagreement is 0.0%."""
    recs = [
        # Same answer 'C', different wrapping.
        _rec("zai", "mmlu", "mmlu-0000", output="The answer is (C)."),
        _rec("umans", "mmlu", "mmlu-0000", output="C"),
    ]
    md = build_markdown(recs)
    # The mmlu row must show 0.0% canonical disagreement somewhere, NOT 100%.
    assert "0.0%" in md
    assert "100.0%" not in md  # canonical disagreement must not be inflated


def test_gsm8k_parity_uses_numeric_extraction():
    """GSM8K canonical comparison extracts the numeric answer, not raw text."""
    recs = [
        _rec("zai", "gsm8k", "gsm8k-0000", output="The total is 5 apples.\n#### 5"),
        _rec("umans", "gsm8k", "gsm8k-0000", output="#### 5"),
    ]
    md = build_markdown(recs)
    # Both extract to 5 -> canonical disagreement must be 0.0%.
    assert "0.0%" in md


def test_subscription_provider_reports_flat_fee_and_tier_table():
    recs = [
        _rec("zai", "mmlu", "mmlu-0000", output="A"),
        _rec("umans", "mmlu", "mmlu-0000", output="A"),
    ]
    providers_pricing = {
        "zai": Pricing(mode="metered", input=1.4, output=4.4),
        "umans": Pricing(mode="subscription", monthly_usd=50.0),
    }
    usage_tiers = {"maximized": UsageTier(tokens_per_day_min=100_000_000, tokens_per_day_max=300_000_000)}
    md = build_markdown(recs, providers_pricing, usage_tiers)
    assert "flat $50.00/mo (subscription" in md
    assert "Effective Cost by Usage Tier" in md
    assert "maximized" in md
