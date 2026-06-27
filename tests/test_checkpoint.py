import json
from glmsbench.checkpoint import Checkpoint
from glmsbench.models import RequestRecord, Timing, RateStats


def _rec(item_id="mmlu-0000", provider="zai", pass_type="parity", idx=0):
    return RequestRecord(
        provider=provider, suite="mmlu", item_id=item_id,
        pass_type=pass_type, pass_index=idx, prompt="p", output="A",
        timing=Timing(), rate=RateStats(), cost_usd=0.01, http_status=200, error=None,
    )


def test_checkpoint_writes_and_reads(tmp_path):
    cp = Checkpoint(tmp_path / "ck.jsonl")
    cp.append(_rec("mmlu-0000", "zai", "parity", 0))
    cp.append(_rec("mmlu-0001", "zai", "parity", 0))
    done = cp.completed_keys()
    assert ("zai", "mmlu", "mmlu-0000", "parity", 0) in done
    assert ("umans", "mmlu", "mmlu-0000", "parity", 0) not in done
    assert len(done) == 2


def test_checkpoint_resume_skips(tmp_path):
    path = tmp_path / "ck.jsonl"
    cp = Checkpoint(path)
    cp.append(_rec("mmlu-0000", "zai", "parity", 0))
    cp2 = Checkpoint(path)
    assert cp2.is_done("zai", "mmlu", "mmlu-0000", "parity", 0) is True
    assert cp2.is_done("umans", "mmlu", "mmlu-0000", "parity", 0) is False
