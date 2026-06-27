from __future__ import annotations
import random

MAX_BACKOFF_MS = 30_000
# Status codes that warrant a retry.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def should_retry(status: int, attempt: int, max_attempts: int) -> bool:
    """True if we should retry given current status and attempt count."""
    if attempt >= max_attempts:
        return False
    return status in RETRYABLE_STATUS


def backoff_ms(attempt: int, base_ms: int, jitter: bool = True) -> float:
    """Exponential backoff (ms), capped at MAX_BACKOFF_MS, with optional jitter."""
    wait = min(base_ms * (2 ** (attempt - 1)), MAX_BACKOFF_MS)
    if jitter:
        wait = random.uniform(0, wait)
    return wait
