import asyncio
import pytest
from glmsbench.models import RequestRecord, Timing, RateStats


class FakeClient:
    """Deterministic fake provider client for runner tests."""
    def __init__(self, name, base_url="x", api_key="k", model_id="m", **kw):
        self.name = name
        self.outputs = {"zai": "A", "umans": "A"}

    async def chat(self, prompt, temperature=0.0, max_tokens=5, seed=42, top_p=1.0, reasoning_effort=None):
        from glmsbench.providers.client import ChatResult
        text = self.outputs[self.name]
        return ChatResult(
            text=text,
            timing=Timing(ttft_ms=100, e2e_ms=200, output_tokens=1, input_tokens=5),
            rate=RateStats(),
            http_status=200, error=None,
        )

    async def aclose(self):
        pass


class FakeLoader:
    name = "mmlu"
    max_tokens = 5

    def load(self, n):
        from glmsbench.datasets.base import DatasetItem
        return [DatasetItem(id=f"mmlu-{i:04d}", prompt="Q?", gold="A", meta={}) for i in range(n)]


async def test_runner_produces_parity_and_latency_records(tmp_path, monkeypatch):
    from glmsbench import runner as runner_mod

    monkeypatch.setattr(runner_mod, "ProviderClient", FakeClient)
    monkeypatch.setattr(runner_mod, "get_loader", lambda name: FakeLoader())

    # minimal config object
    from glmsbench.config import Config, Generation, Repeats, Pricing, ProviderConfig, Rate
    cfg = Config(
        model="glm-5.2",
        generation=Generation(temperature=0.0, max_tokens=5, seed=42),
        repeats=Repeats(parity=1, latency=2),
        suites=["mmlu"],
        suite_sizes={"mmlu": 2},
        providers={
            "zai": ProviderConfig(base_url="x", api_key_env="X", model_id="m",
                                  pricing=Pricing(input=1, output=1), rate=Rate(concurrency=2)),
            "umans": ProviderConfig(base_url="x", api_key_env="X", model_id="m",
                                    pricing=Pricing(input=1, output=1), rate=Rate(concurrency=2)),
        },
    )
    records = await runner_mod.run_benchmark(cfg, out_dir=str(tmp_path), checkpoint_path=str(tmp_path / "ck.jsonl"))
    # 2 items * 2 providers * (1 parity + 2 latency) = 12
    assert len(records) == 12
    parity = [r for r in records if r.pass_type == "parity"]
    latency = [r for r in records if r.pass_type == "latency"]
    assert len(parity) == 4   # 2 items * 2 providers
    assert len(latency) == 8  # 2 items * 2 providers * 2 repeats


async def test_runner_fail_fast_stops_provider_after_429(tmp_path, monkeypatch):
    from glmsbench import runner as runner_mod
    from glmsbench.providers.client import ChatResult

    calls = {"zai": 0, "umans": 0}

    class FailFastClient(FakeClient):
        async def chat(self, prompt, temperature=0.0, max_tokens=5, seed=42, top_p=1.0, reasoning_effort=None):
            calls[self.name] += 1
            if self.name == "umans":
                return ChatResult("", Timing(), RateStats(n_throttled=1), http_status=429, error="HTTP 429")
            return await super().chat(prompt, temperature, max_tokens, seed, top_p, reasoning_effort)

    monkeypatch.setattr(runner_mod, "ProviderClient", FailFastClient)
    monkeypatch.setattr(runner_mod, "get_loader", lambda name: FakeLoader())

    from glmsbench.config import Config, Generation, Repeats, Pricing, ProviderConfig, Rate, RetryConfig
    cfg = Config(
        model="glm-5.2",
        generation=Generation(temperature=0.0, max_tokens=5, seed=42),
        repeats=Repeats(parity=1, latency=2),
        suites=["mmlu"],
        suite_sizes={"mmlu": 3},
        providers={
            "zai": ProviderConfig(base_url="x", api_key_env="X", model_id="m",
                                  pricing=Pricing(input=1, output=1), rate=Rate(concurrency=2)),
            "umans": ProviderConfig(base_url="x", api_key_env="X", model_id="m",
                                    pricing=Pricing(input=1, output=1),
                                    rate=Rate(concurrency=1, retry=RetryConfig(max_attempts=1), fail_fast_statuses=[429, 403])),
        },
    )
    records = await runner_mod.run_benchmark(cfg, out_dir=str(tmp_path), checkpoint_path=str(tmp_path / "ck.jsonl"))

    assert calls["zai"] == 9       # 3 items * (1 parity + 2 latency)
    assert calls["umans"] == 1     # stopped immediately after first 429
    umans_records = [r for r in records if r.provider == "umans"]
    assert len(umans_records) == 1
    assert umans_records[0].http_status == 429
