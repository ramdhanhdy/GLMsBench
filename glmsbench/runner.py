from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional
from .config import Config
from .models import RequestRecord
from .checkpoint import Checkpoint
from .providers.client import ProviderClient
from .providers.ratelimit import RateLimiter
from .datasets.base import Loader


def get_loader(name: str) -> Loader:
    from .datasets.mmlu import MMLULoader
    from .datasets.gsm8k import GSM8KLoader
    from .datasets.arc import ARCLoader
    from .datasets.humaneval import HumanEvalLoader
    return {
        "mmlu": MMLULoader,
        "gsm8k": GSM8KLoader,
        "arc": ARCLoader,
        "humaneval": HumanEvalLoader,
    }[name]()


@dataclass(frozen=True)
class _RequestSpec:
    provider_name: str
    suite: str
    item: object
    pass_type: str
    pass_index: int
    max_tokens: int


async def run_benchmark(
    cfg: Config, out_dir: str, checkpoint_path: str, suites: Optional[list[str]] = None,
) -> list[RequestRecord]:
    suites = suites or cfg.suites
    os.makedirs(out_dir, exist_ok=True)

    # Build clients + limiters per provider
    clients: dict[str, ProviderClient] = {}
    limiters: dict[str, RateLimiter] = {}
    for name, p in cfg.providers.items():
        clients[name] = ProviderClient(
            name=name,
            base_url=p.base_url, api_key=p.api_key, model_id=p.model_id,
            max_attempts=p.rate.retry.max_attempts, base_backoff_ms=p.rate.retry.base_backoff_ms,
        )
        limiters[name] = RateLimiter(p.rate.concurrency, p.rate.rpm, p.rate.tpm)

    checkpoint = Checkpoint(checkpoint_path)
    all_records: list[RequestRecord] = []

    # Load existing checkpointed records so reporters have the full set
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                all_records.append(_record_from_dict(d))

    for suite in suites:
        loader = get_loader(suite)
        n = cfg.suite_sizes.get(suite, 100)
        items = loader.load(n)
        max_tokens = loader.max_tokens

        normal_tasks = []
        fail_fast_specs: dict[str, list[_RequestSpec]] = {}
        for provider_name in cfg.providers:
            provider_cfg = cfg.providers[provider_name]
            for item in items:
                # parity pass
                if not checkpoint.is_done(provider_name, suite, item.id, "parity", 0):
                    spec = _RequestSpec(provider_name, suite, item, "parity", 0, max_tokens)
                    if provider_cfg.rate.fail_fast_statuses:
                        fail_fast_specs.setdefault(provider_name, []).append(spec)
                    else:
                        normal_tasks.append(_run_one(
                            clients[provider_name], limiters[provider_name],
                            provider_name, suite, item, "parity", 0,
                            cfg.generation.temperature, max_tokens, cfg.generation.seed,
                            cfg.providers[provider_name].pricing, cfg.generation.reasoning_effort,
                        ))
                # latency repeats
                for k in range(cfg.repeats.latency):
                    if not checkpoint.is_done(provider_name, suite, item.id, "latency", k):
                        spec = _RequestSpec(provider_name, suite, item, "latency", k, max_tokens)
                        if provider_cfg.rate.fail_fast_statuses:
                            fail_fast_specs.setdefault(provider_name, []).append(spec)
                        else:
                            normal_tasks.append(_run_one(
                                clients[provider_name], limiters[provider_name],
                                provider_name, suite, item, "latency", k,
                                cfg.generation.temperature, max_tokens, cfg.generation.seed,
                                cfg.providers[provider_name].pricing, cfg.generation.reasoning_effort,
                            ))

        results = await asyncio.gather(*normal_tasks)
        for r in results:
            if r is not None:
                checkpoint.append(r)
                all_records.append(r)

        for provider_name, specs in fail_fast_specs.items():
            provider_cfg = cfg.providers[provider_name]
            fail_fast_statuses = set(provider_cfg.rate.fail_fast_statuses)
            for spec in specs:
                r = await _run_one(
                    clients[provider_name], limiters[provider_name],
                    provider_name, spec.suite, spec.item, spec.pass_type, spec.pass_index,
                    cfg.generation.temperature, spec.max_tokens, cfg.generation.seed,
                    provider_cfg.pricing, cfg.generation.reasoning_effort,
                )
                if r is not None:
                    checkpoint.append(r)
                    all_records.append(r)
                if r is not None and r.http_status in fail_fast_statuses:
                    break
                if provider_cfg.rate.request_delay_s > 0:
                    await asyncio.sleep(provider_cfg.rate.request_delay_s)

    for c in clients.values():
        await c.aclose()
    return all_records


def _compute_cost(record: RequestRecord, pricing) -> Optional[float]:
    """Per-request cost for metered pricing; None for flat-fee subscriptions,
    where cost isn't attributable to a single request (see UsageTier-based
    effective-cost reporting instead)."""
    if pricing.mode == "subscription":
        return None
    if not record.timing:
        return 0.0
    return (record.timing.input_tokens / 1_000_000) * pricing.input + \
           (record.timing.output_tokens / 1_000_000) * pricing.output


async def _run_one(client, limiter, provider, suite, item, pass_type, pass_index,
                   temperature, max_tokens, seed, pricing, reasoning_effort=None) -> Optional[RequestRecord]:
    async with limiter.acquire():
        result = await client.chat(
            item.prompt, temperature=temperature, max_tokens=max_tokens, seed=seed,
            reasoning_effort=reasoning_effort,
        )
    record = RequestRecord(
        provider=provider, suite=suite, item_id=item.id, pass_type=pass_type,
        pass_index=pass_index, prompt=item.prompt, output=result.text,
        timing=result.timing, rate=result.rate, http_status=result.http_status,
        error=result.error,
    )
    record.cost_usd = _compute_cost(record, pricing)
    return record


def _record_from_dict(d: dict) -> RequestRecord:
    from .models import Timing, RateStats
    timing = Timing(**(d.get("timing") or {}))
    rate = RateStats(**(d.get("rate") or {}))
    return RequestRecord(
        provider=d["provider"], suite=d["suite"], item_id=d["item_id"],
        pass_type=d["pass_type"], pass_index=d["pass_index"], prompt=d["prompt"],
        output=d.get("output", ""), timing=timing, rate=rate,
        cost_usd=d.get("cost_usd"), http_status=d.get("http_status"), error=d.get("error"),
    )
