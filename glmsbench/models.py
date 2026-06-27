from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Timing:
    """Raw timing captured per request via streaming."""
    ttft_ms: Optional[float] = None          # time to first token
    inter_token_latency_ms: Optional[float] = None  # median of per-token gaps
    e2e_ms: Optional[float] = None           # request sent -> last token
    output_tokens: int = 0
    input_tokens: int = 0
    stop_reason: Optional[str] = None


@dataclass
class RateStats:
    """Rate-limit behavior over a request's lifecycle (after retries)."""
    n_attempts: int = 1
    n_throttled: int = 0        # count of 429s encountered
    total_backoff_ms: float = 0.0
    had_backoff: bool = False   # any backoff occurred -> excluded from clean latency


@dataclass
class RequestRecord:
    """One completed (or failed) request to one provider."""
    provider: str
    suite: str
    item_id: str
    pass_type: str              # "parity" | "latency"
    pass_index: int             # 0 for parity, 0..N-1 for latency repeats
    prompt: str
    output: str = ""
    timing: Optional[Timing] = None
    rate: Optional[RateStats] = None
    cost_usd: Optional[float] = None
    http_status: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.http_status == 200 and self.timing is not None
