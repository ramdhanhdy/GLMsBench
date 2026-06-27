from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ConfigError(Exception):
    pass


class Generation(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int = 42


class Repeats(BaseModel):
    parity: int = 1
    latency: int = 5


class Pricing(BaseModel):
    input: float
    output: float


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


def load_config(path: str) -> Config:
    import yaml
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
