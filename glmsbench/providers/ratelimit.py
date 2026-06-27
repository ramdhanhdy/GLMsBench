from __future__ import annotations
import asyncio
import time
from typing import Optional


class RateLimiter:
    """Per-provider concurrency cap + optional rpm leaky-bucket.

    tpm limiting is intentionally omitted at acquire() level — token accounting
    needs the actual request size which isn't known until the request runs.
    tpm is enforced via retry-on-429 instead (see client.py).

    Usage: ``async with limiter.acquire(): ...`` — acquire() returns an async
    context manager (NOT a coroutine), so it composes directly with ``async with``.
    """

    def __init__(self, concurrency: int, rpm: Optional[int] = None, tpm: Optional[int] = None):
        self._sem = asyncio.Semaphore(concurrency)
        self._rpm = rpm
        self._tpm = tpm
        self._min_interval = (60.0 / rpm) if rpm else 0.0
        self._last_release_time = 0.0
        self._tick_lock = asyncio.Lock()

    def acquire(self) -> "_AcquireContext":
        return _AcquireContext(self)


class _AcquireContext:
    """Async context manager: acquires the semaphore + applies rpm pacing on
    __aenter__, releases the semaphore on __aexit__."""

    def __init__(self, owner: RateLimiter):
        self._owner = owner

    async def __aenter__(self) -> "_AcquireContext":
        await self._owner._sem.acquire()
        if self._owner._min_interval > 0:
            async with self._owner._tick_lock:
                now = time.monotonic()
                wait = self._owner._last_release_time + self._owner._min_interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
                self._owner._last_release_time = time.monotonic()
        return self

    async def __aexit__(self, *exc):
        self._owner._sem.release()
