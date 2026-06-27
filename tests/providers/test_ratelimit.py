import asyncio
import time
import pytest
from glmsbench.providers.ratelimit import RateLimiter


async def test_concurrency_semaphore_limits_inflight():
    limiter = RateLimiter(concurrency=2)
    inflight = 0
    peak = 0
    lock = asyncio.Lock()

    async def task():
        nonlocal inflight, peak
        async with limiter.acquire():
            async with lock:
                inflight += 1
                peak = max(peak, inflight)
            await asyncio.sleep(0.05)
            async with lock:
                inflight -= 1

    await asyncio.gather(*[task() for _ in range(6)])
    assert peak <= 2


async def test_rpm_throttles():
    # 60 rpm = 1 per second. Two rapid requests -> second one waits ~1s.
    limiter = RateLimiter(concurrency=10, rpm=60)
    t0 = time.monotonic()
    async with limiter.acquire():
        pass
    async with limiter.acquire():
        pass
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.9   # at least ~1s due to rpm gate
