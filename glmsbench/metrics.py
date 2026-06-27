from __future__ import annotations
import statistics
from typing import Optional
from .models import RequestRecord, Timing


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _series(records: list[RequestRecord], attr: str) -> dict:
    vals = [getattr(r.timing, attr) for r in records if r.timing and getattr(r.timing, attr) is not None]
    return {
        "n": len(vals),
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "p95": _percentile(vals, 0.95),
    }


def latency_stats(records: list[RequestRecord], provider: str, suite: str) -> dict:
    """Compute clean vs effective latency stats for a provider+suite.

    clean = latency repeats that succeeded with no backoff.
    effective = all latency repeats (backoff included, via e2e).
    """
    base = [r for r in records if r.provider == provider and r.suite == suite
            and r.pass_type == "latency" and r.ok]
    clean = [r for r in base if not (r.rate and r.rate.had_backoff)]
    effective = base  # e2e already includes backoff wall-clock

    def series_for(recs):
        return {
            "ttft_ms": _series(recs, "ttft_ms"),
            "inter_token_latency_ms": _series(recs, "inter_token_latency_ms"),
            "e2e_ms": _series(recs, "e2e_ms"),
        }

    return {
        "clean": {**series_for(clean), "n": len(clean)},
        "effective": {**series_for(effective), "n": len(effective)},
    }


def cost_total(records: list[RequestRecord], provider: str) -> float:
    return sum(r.cost_usd or 0.0 for r in records if r.provider == provider and r.ok)


def tokens_per_sec(t: Timing) -> Optional[float]:
    if t.e2e_ms is None or t.ttft_ms is None or t.e2e_ms == t.ttft_ms:
        return None
    return t.output_tokens / ((t.e2e_ms - t.ttft_ms) / 1000.0)
