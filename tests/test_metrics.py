from glmsbench.models import RequestRecord, Timing, RateStats
from glmsbench.metrics import latency_stats, cost_total, tokens_per_sec


def _record(provider, suite, ttft, e2e, out_tok=10, in_tok=5, had_backoff=False, ok=True, cost=0.01):
    return RequestRecord(
        provider=provider, suite=suite, item_id="x", pass_type="latency", pass_index=0,
        prompt="p", output="o",
        timing=Timing(ttft_ms=ttft, e2e_ms=e2e, output_tokens=out_tok, input_tokens=in_tok),
        rate=RateStats(had_backoff=had_backoff),
        cost_usd=cost,
        http_status=200 if ok else 500,
        error=None if ok else "boom",
    )


def test_latency_stats_clean_only():
    recs = [
        _record("zai", "mmlu", ttft=100, e2e=200),
        _record("zai", "mmlu", ttft=120, e2e=220, had_backoff=True),
    ]
    stats = latency_stats(recs, provider="zai", suite="mmlu")
    # clean latency excludes the backoff one -> only one sample (100/200)
    assert stats["clean"]["n"] == 1
    assert stats["clean"]["ttft_ms"]["median"] == 100
    assert stats["effective"]["n"] == 2


def test_cost_total():
    recs = [_record("zai", "mmlu", 100, 200, cost=0.02), _record("zai", "mmlu", 100, 200, cost=0.03)]
    assert cost_total(recs, provider="zai") == 0.05


def test_tokens_per_sec():
    t = Timing(ttft_ms=100, e2e_ms=1100, output_tokens=100, input_tokens=5)
    # 100 tokens over (1100-100)ms = 100 tokens/sec
    assert tokens_per_sec(t) == 100.0
