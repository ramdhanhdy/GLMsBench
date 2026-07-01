from __future__ import annotations
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ConfigError(Exception):
    pass


class Generation(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int = 42
    # GLM-5.2 reasons by default on both providers (hidden reasoning_content
    # counts against max_tokens before any visible answer). "low" keeps a
    # short reasoning trace instead of paying for deep/max reasoning on
    # simple MCQ/numeric tasks. "none" disables reasoning entirely.
    # Comparing reasoning levels head-to-head is a deferred follow-up (see
    # spec Open Items) — for now this is a single fixed value per run.
    reasoning_effort: Optional[str] = "low"


class Repeats(BaseModel):
    parity: int = 1
    latency: int = 5


class Pricing(BaseModel):
    """Provider pricing, either metered (pay-per-token) or a flat-fee
    subscription. Many real-world coding plans (Z.ai's Lite/Pro, umans'
    tiers) are subscriptions with usage quotas, not per-token metering, so
    per-request cost isn't meaningful for them — see UsageTier below for
    how subscription cost is instead expressed as an effective $/1M-token
    range under an assumed usage intensity.
    """
    mode: Literal["metered", "subscription"] = "metered"
    input: Optional[float] = None
    output: Optional[float] = None
    monthly_usd: Optional[float] = None

    @model_validator(mode="after")
    def _check_required_fields(self):
        if self.mode == "metered" and (self.input is None or self.output is None):
            raise ConfigError("pricing.mode 'metered' requires 'input' and 'output'")
        if self.mode == "subscription" and self.monthly_usd is None:
            raise ConfigError("pricing.mode 'subscription' requires 'monthly_usd'")
        return self


class UsageTier(BaseModel):
    """A named usage-intensity band (tokens/day) used to translate a flat
    subscription fee into a comparable effective $/1M-token range."""
    tokens_per_day_min: float
    tokens_per_day_max: float

    @model_validator(mode="after")
    def _check_range(self):
        if self.tokens_per_day_min <= 0:
            raise ConfigError("usage_tiers.tokens_per_day_min must be > 0")
        if self.tokens_per_day_max < self.tokens_per_day_min:
            raise ConfigError("usage_tiers.tokens_per_day_max must be >= tokens_per_day_min")
        return self


class RetryConfig(BaseModel):
    max_attempts: int = 5
    base_backoff_ms: int = 1000


class Rate(BaseModel):
    concurrency: int = 4
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    retry: RetryConfig = Field(default_factory=RetryConfig)


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str
    model_id: str
    pricing: Pricing
    rate: Rate = Field(default_factory=Rate)
    # resolved at load time
    api_key: str = ""

    @model_validator(mode="after")
    def _check_positive(self):
        if self.rate.concurrency < 1:
            raise ConfigError("concurrency must be >= 1")
        return self


class Config(BaseModel):
    model: str
    generation: Generation
    repeats: Repeats
    suites: list[str]
    suite_sizes: dict[str, int] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig]
    usage_tiers: dict[str, UsageTier] = Field(default_factory=dict)


def load_config(path: str) -> Config:
    import yaml
    from dotenv import load_dotenv

    # Populate os.environ from a .env file if present (real env vars take
    # precedence — override=False is the dotenv default). Harmless no-op if
    # no .env file exists.
    load_dotenv()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        cfg = Config(**raw)
    except Exception as e:
        raise ConfigError(f"invalid config: {e}") from e

    if len(cfg.providers) != 2:
        raise ConfigError(f"config must define exactly 2 providers, got {len(cfg.providers)}")

    for name, p in cfg.providers.items():
        key = os.environ.get(p.api_key_env)
        if not key:
            raise ConfigError(
                f"provider '{name}': environment variable '{p.api_key_env}' is not set"
            )
        p.api_key = key

    if cfg.repeats.parity < 1 or cfg.repeats.latency < 1:
        raise ConfigError("repeats.parity and repeats.latency must be >= 1")

    return cfg
