import pytest
from glmsbench.providers.retry import should_retry, backoff_ms


def test_retry_on_429():
    assert should_retry(429, attempt=1, max_attempts=5) is True


def test_retry_on_503():
    assert should_retry(503, attempt=1, max_attempts=5) is True


def test_no_retry_on_400():
    assert should_retry(400, attempt=1, max_attempts=5) is False


def test_no_retry_after_max_attempts():
    assert should_retry(429, attempt=5, max_attempts=5) is False


def test_backoff_increases_with_jitter_bounds():
    # without jitter: base grows exponentially
    b1 = backoff_ms(attempt=1, base_ms=1000, jitter=False)
    b2 = backoff_ms(attempt=2, base_ms=1000, jitter=False)
    b3 = backoff_ms(attempt=3, base_ms=1000, jitter=False)
    assert b1 < b2 < b3
    assert b1 == 1000 and b2 == 2000 and b3 == 4000


def test_backoff_caps_at_30s():
    b = backoff_ms(attempt=10, base_ms=1000, jitter=False)
    assert b <= 30000


def test_backoff_with_jitter_in_range():
    for _ in range(20):
        b = backoff_ms(attempt=2, base_ms=1000, jitter=True)
        assert 0 <= b <= 4000
