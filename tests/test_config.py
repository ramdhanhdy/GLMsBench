import os
import pytest
from glmsbench.config import load_config, ConfigError, Pricing, UsageTier


def _set_env(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "k1")
    monkeypatch.setenv("UMANS_API_KEY", "k2")


def test_loads_valid_config(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
model: glm-5.2
generation:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 1024
  seed: 42
repeats:
  parity: 1
  latency: 5
suites: [mmlu]
suite_sizes: {mmlu: 5}
providers:
  zai:
    base_url: https://api.z.ai/v1
    api_key_env: ZAI_API_KEY
    model_id: glm-5.2
    pricing: {input: 1.40, output: 4.40}
    rate: {concurrency: 2}
  umans:
    base_url: https://example.com/v1
    api_key_env: UMANS_API_KEY
    model_id: glm-5.2
    pricing: {input: 1.0, output: 3.0}
    rate: {concurrency: 6}
"""
    )
    c = load_config(str(cfg))
    assert c.model == "glm-5.2"
    assert len(c.providers) == 2
    assert c.providers["zai"].api_key == "k1"
    assert c.providers["umans"].api_key == "k2"


def test_rejects_three_providers(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
model: glm-5.2
generation: {temperature: 0.0, max_tokens: 5, seed: 1}
repeats: {parity: 1, latency: 1}
suites: [mmlu]
providers:
  a: {base_url: "https://a", api_key_env: ZAI_API_KEY, model_id: m, pricing: {input: 1, output: 1}}
  b: {base_url: "https://b", api_key_env: UMANS_API_KEY, model_id: m, pricing: {input: 1, output: 1}}
  c: {base_url: "https://c", api_key_env: ZAI_API_KEY, model_id: m, pricing: {input: 1, output: 1}}
"""
    )
    with pytest.raises(ConfigError, match="exactly 2"):
        load_config(str(cfg))


def test_rejects_missing_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
model: glm-5.2
generation: {temperature: 0.0, max_tokens: 5, seed: 1}
repeats: {parity: 1, latency: 1}
suites: [mmlu]
providers:
  zai:
    base_url: "https://a"
    api_key_env: DOES_NOT_EXIST
    model_id: m
    pricing: {input: 1, output: 1}
  umans:
    base_url: "https://b"
    api_key_env: UMANS_API_KEY
    model_id: m
    pricing: {input: 1, output: 1}
"""
    )
    monkeypatch.setenv("UMANS_API_KEY", "k2")
    with pytest.raises(ConfigError, match="environment variable"):
        load_config(str(cfg))


def test_pricing_metered_defaults_and_requires_input_output():
    p = Pricing(input=1.0, output=2.0)
    assert p.mode == "metered"
    with pytest.raises(ConfigError, match="requires 'input' and 'output'"):
        Pricing(mode="metered", output=2.0)


def test_pricing_subscription_requires_monthly_usd():
    p = Pricing(mode="subscription", monthly_usd=50.0)
    assert p.monthly_usd == 50.0
    with pytest.raises(ConfigError, match="requires 'monthly_usd'"):
        Pricing(mode="subscription")


def test_usage_tier_validates_range():
    UsageTier(tokens_per_day_min=50_000_000, tokens_per_day_max=90_000_000)  # ok
    with pytest.raises(ConfigError, match="tokens_per_day_max"):
        UsageTier(tokens_per_day_min=90_000_000, tokens_per_day_max=50_000_000)
    with pytest.raises(ConfigError, match="tokens_per_day_min"):
        UsageTier(tokens_per_day_min=0, tokens_per_day_max=50_000_000)


def test_loads_config_with_subscription_pricing_and_usage_tiers(tmp_path, monkeypatch):
    _set_env(monkeypatch)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
model: glm-5.2
generation: {temperature: 0.0, max_tokens: 5, seed: 1}
repeats: {parity: 1, latency: 1}
suites: [mmlu]
usage_tiers:
  early_setup: {tokens_per_day_min: 50000000, tokens_per_day_max: 90000000}
  maximized: {tokens_per_day_min: 100000000, tokens_per_day_max: 300000000}
providers:
  zai:
    base_url: "https://a"
    api_key_env: ZAI_API_KEY
    model_id: m
    pricing: {mode: subscription, monthly_usd: 16}
  umans:
    base_url: "https://b"
    api_key_env: UMANS_API_KEY
    model_id: m
    pricing: {mode: subscription, monthly_usd: 50}
"""
    )
    c = load_config(str(cfg))
    assert c.providers["zai"].pricing.mode == "subscription"
    assert c.providers["zai"].pricing.monthly_usd == 16
    assert set(c.usage_tiers) == {"early_setup", "maximized"}
    assert c.usage_tiers["maximized"].tokens_per_day_max == 300_000_000
