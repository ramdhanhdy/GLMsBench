from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from ..models import Timing, RateStats
from .retry import should_retry, backoff_ms


@dataclass
class ChatResult:
    text: str
    timing: Timing
    rate: RateStats
    http_status: int
    error: Optional[str] = None


class ProviderClient:
    """Async OpenAI-compatible streaming client.

    Records honest client-side timing (TTFT, inter-token latency, e2e) and
    token usage. Retries 429/5xx with exponential backoff; records all
    throttling/backoff in RateStats so it can be reported and excluded from
    clean latency.

    An injectable ``_transport`` lets tests run without the network.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_s: float = 120.0,
        max_attempts: int = 5,
        base_backoff_ms: int = 1000,
        _transport: Optional[httpx.BaseTransport] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._max_attempts = max_attempts
        self._base_backoff_ms = base_backoff_ms
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=_transport,
        )

    async def aclose(self):
        await self._client.aclose()

    async def chat(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        top_p: float = 1.0,
    ) -> ChatResult:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        rate = RateStats()
        last_status = 0
        last_error: Optional[str] = None

        for attempt in range(1, self._max_attempts + 1):
            rate.n_attempts = attempt
            try:
                t_start = time.monotonic()
                async with self._client.stream(
                    "POST", url, json=payload, headers=headers
                ) as resp:
                    last_status = resp.status_code
                    if resp.status_code != 200:
                        # drain body for error text
                        body = await resp.aread()
                        last_error = f"HTTP {resp.status_code}: {body[:200]!r}"
                        if should_retry(resp.status_code, attempt, self._max_attempts):
                            if resp.status_code == 429:
                                rate.n_throttled += 1
                            retry_after = self._retry_after_ms(resp)
                            wait = retry_after or backoff_ms(attempt, self._base_backoff_ms)
                            rate.total_backoff_ms += wait
                            rate.had_backoff = True
                            await asyncio.sleep(wait / 1000.0)
                            continue
                        return ChatResult("", Timing(), rate, resp.status_code, last_error)

                    text, timing = await self._consume_stream(resp, t_start)
                    return ChatResult(text, timing, rate, 200, None)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_status = 0
                last_error = f"transport: {type(e).__name__}: {e}"
                if attempt < self._max_attempts:
                    wait = backoff_ms(attempt, self._base_backoff_ms)
                    rate.total_backoff_ms += wait
                    rate.had_backoff = True
                    await asyncio.sleep(wait / 1000.0)
                    continue
                return ChatResult("", Timing(), rate, 0, last_error)

        return ChatResult("", Timing(), rate, last_status, last_error or "exhausted retries")

    async def _consume_stream(self, resp, t_start: float) -> tuple[str, Timing]:
        text_parts: list[str] = []
        ttft: Optional[float] = None
        token_times: list[float] = []
        usage = {}
        finish_reason = None

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                tok = delta.get("content")
                if tok:
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - t_start
                    token_times.append(now)
                    text_parts.append(tok)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

        text = "".join(text_parts)
        t_end = time.monotonic()

        inter = None
        if len(token_times) >= 2:
            gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            inter = sorted(gaps)[len(gaps) // 2]  # median

        timing = Timing(
            ttft_ms=(ttft * 1000.0) if ttft is not None else None,
            inter_token_latency_ms=(inter * 1000.0) if inter is not None else None,
            e2e_ms=(t_end - t_start) * 1000.0,
            output_tokens=usage.get("completion_tokens", len(text_parts)),
            input_tokens=usage.get("prompt_tokens", 0),
            stop_reason=finish_reason,
        )
        return text, timing

    @staticmethod
    def _retry_after_ms(resp) -> Optional[float]:
        ra = resp.headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra) * 1000.0
        except ValueError:
            return None
