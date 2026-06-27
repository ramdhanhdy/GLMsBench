import json
from glmsbench.models import RequestRecord, Timing, RateStats
from glmsbench.reporting.json_out import build_report


def _rec(provider, suite, item_id, pass_type="parity", output="A", correct=True, cost=0.01):
    return RequestRecord(
        provider=provider, suite=suite, item_id=item_id, pass_type=pass_type, pass_index=0,
        prompt="p", output=output,
        timing=Timing(ttft_ms=100, e2e_ms=200, output_tokens=1, input_tokens=5),
        rate=RateStats(), cost_usd=cost, http_status=200, error=None,
    )


def test_build_report_structure(tmp_path):
    recs = [
        _rec("zai", "mmlu", "mmlu-0000"),
        _rec("umans", "mmlu", "mmlu-0000"),
    ]
    report = build_report(recs, config_dump={"model": "glm-5.2"})
    assert "run_meta" in report
    assert "requests" in report
    assert len(report["requests"]) == 2
    assert report["run_meta"]["config"]["model"] == "glm-5.2"


def test_report_has_aggregates_with_canonical_parity():
    """Spec §9: JSON report must carry an aggregates block. The parity
    aggregate must use canonical comparison (same answer, different prose ->
    0 disagreement). Regression guard for the §6.2 bug."""
    recs = [
        _rec("zai", "mmlu", "mmlu-0000", output="The answer is (C)."),
        _rec("umans", "mmlu", "mmlu-0000", output="C"),
    ]
    report = build_report(recs)
    assert "aggregates" in report
    parity = report["aggregates"]["by_suite"]["mmlu"]["parity"]
    assert parity["canonical_disagreement_rate"] == 0.0  # canonical agrees
    assert parity["raw_text_agreement_rate"] == 0.0       # raw text differs
    assert "cost" in report["aggregates"]
