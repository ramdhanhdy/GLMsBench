import pytest
import httpx
import json


def _build_stream_chunks():
    """Yield OpenAI-style SSE chunks as raw bytes."""
    parts = []
    for tok in ["Hello", " world"]:
        parts.append(b"data: " + json.dumps({
            "choices": [{"delta": {"content": tok}}]
        }).encode() + b"\n\n")
    # final chunk with usage
    parts.append(b"data: " + json.dumps({
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    }).encode() + b"\n\n")
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


async def test_client_parses_stream_and_timing():
    from glmsbench.providers.client import ProviderClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_build_stream_chunks(),
        )

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, _transport=transport_mock,
    )
    try:
        result = await client.chat("hi", temperature=0.0, max_tokens=10, seed=1)
    finally:
        await client.aclose()
    assert result.text == "Hello world"
    assert result.timing.output_tokens == 2
    assert result.timing.input_tokens == 5
    assert result.timing.ttft_ms is not None and result.timing.ttft_ms >= 0
    assert result.timing.e2e_ms is not None and result.timing.e2e_ms >= result.timing.ttft_ms
    assert result.rate.n_attempts == 1
    assert result.rate.n_throttled == 0
    assert result.http_status == 200


async def test_client_sends_reasoning_effort_when_set():
    from glmsbench.providers.client import ProviderClient

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_build_stream_chunks(),
        )

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, _transport=transport_mock,
    )
    try:
        await client.chat("hi", temperature=0.0, max_tokens=10, seed=1, reasoning_effort="low")
    finally:
        await client.aclose()
    assert captured["payload"]["reasoning_effort"] == "low"


async def test_client_omits_reasoning_effort_when_none():
    from glmsbench.providers.client import ProviderClient

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_build_stream_chunks(),
        )

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, _transport=transport_mock,
    )
    try:
        await client.chat("hi", temperature=0.0, max_tokens=10, seed=1)
    finally:
        await client.aclose()
    assert "reasoning_effort" not in captured["payload"]


async def test_client_captures_reasoning_tokens_from_usage():
    from glmsbench.providers.client import ProviderClient

    def handler(request: httpx.Request) -> httpx.Response:
        parts = [
            b"data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking..."}}]}).encode() + b"\n\n",
            b"data: " + json.dumps({"choices": [{"delta": {"content": "C"}}]}).encode() + b"\n\n",
            b"data: " + json.dumps({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5, "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 9},
                },
            }).encode() + b"\n\n",
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"".join(parts),
        )

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, _transport=transport_mock,
    )
    try:
        result = await client.chat("hi", temperature=0.0, max_tokens=100, seed=1, reasoning_effort="low")
    finally:
        await client.aclose()
    assert result.text == "C"
    assert result.timing.reasoning_tokens == 9
    assert result.timing.output_tokens == 10


async def test_client_records_429_then_succeeds():
    from glmsbench.providers.client import ProviderClient

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, content=b"rate limited")
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_build_stream_chunks(),
        )

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, max_attempts=3, base_backoff_ms=1,
        _transport=transport_mock,
    )
    try:
        result = await client.chat("hi", temperature=0.0, max_tokens=10, seed=1)
    finally:
        await client.aclose()
    assert result.http_status == 200
    assert result.text == "Hello world"
    assert result.rate.n_throttled == 1
    assert result.rate.had_backoff is True
    assert result.rate.n_attempts == 2


async def test_client_returns_error_after_exhausted_retries():
    from glmsbench.providers.client import ProviderClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b"rate limited")

    transport_mock = httpx.MockTransport(handler)
    client = ProviderClient(
        base_url="https://x", api_key="k", model_id="m",
        timeout_s=30.0, max_attempts=2, base_backoff_ms=1,
        _transport=transport_mock,
    )
    try:
        result = await client.chat("hi", temperature=0.0, max_tokens=10, seed=1)
    finally:
        await client.aclose()
    assert result.http_status == 429
    assert result.error is not None
    assert result.rate.n_throttled >= 1
