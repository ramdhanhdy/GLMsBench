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
