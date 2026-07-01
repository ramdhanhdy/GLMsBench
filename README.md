# GLMsBench — GLM-5.2 Provider Benchmark

A benchmark harness that compares **GLM-5.2** performance across two inference
providers — [Z.ai](https://z.ai) and [umans](https://umans.ai) — on standard
evaluation suites (MMLU, GSM8K, ARC-Challenge, HumanEval). It measures
latency, output parity, cost, and rate-limit behavior, then reports the
results as JSON + Markdown.

Both providers serve the same underlying model (GLM-5.2), but through
different infrastructure, pricing, and rate limits. This harness answers:
*Are they actually equivalent from a user's perspective?*

## What it measures

- **Latency** — TTFT, inter-token, and end-to-end, reported as p50/p95.
  Two populations are tracked: *clean* (no throttling) for apples-to-apples
  inference speed, and *effective* (including backoff) for real-world
  experience under load.
- **Parity** — cross-provider answer disagreement at temperature=0. The
  headline number for "do Z.ai and umans produce the same output?"
- **Cost** — supports both metered (per-token) and subscription (flat-fee)
  pricing models. Subscription plans are translated into an effective
  $/1M-token range using configurable usage-intensity tiers.
- **Rate limits** — 429s, backoff time, and retry counts per provider,
  treated as a first-class signal rather than noise.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"         # macOS/Linux
cp .env.example .env   # then fill in ZAI_API_KEY / UMANS_API_KEY
```

API keys are loaded from a `.env` file in the project root (auto-loaded by
`load_config`, via `python-dotenv`) — `.env` is gitignored by default.
Real shell environment variables (`export`/`$env:`) still work and take
precedence over `.env` if both are set.

## Usage

### Full benchmark

```bash
.venv/Scripts/python.exe run.py --config harness/config.example.yaml --out results/
```

Flags: `--suite mmlu,gsm8k` (subset), `--out results/`, `--resume`.

### Smoke test (minimal spend)

Runs a tiny sample against live APIs to verify the pipeline end-to-end
before committing budget to a full run:

```bash
.venv/Scripts/python.exe scripts/smoke_test.py --config harness/config.example.yaml
.venv/Scripts/python.exe scripts/smoke_test.py --config harness/config.example.yaml --suite arc --n 3
```

## Configuration

The config file (`harness/config.example.yaml`) defines providers, pricing,
rate limits, and usage tiers. Key fields:

- **`providers`** — `base_url`, `api_key_env`, `model_id`, `pricing`, and
  `rate` (concurrency / rpm / tpm) per provider.
- **`pricing`** — `mode: metered` (per-token `input`/`output` rates) or
  `mode: subscription` (flat `monthly_usd`).
- **`usage_tiers`** — tokens/day bands (e.g. `early_setup`, `maximized`)
  used to compute an effective $/1M-token range for subscription providers.
- **`generation`** — `temperature`, `max_tokens`, `seed`, and
  `reasoning_effort` (GLM-5.2 reasons by default; set to `"none"` to
  disable, `"low"`/`"medium"`/`"high"` to control depth).

## Architecture

```
glmsbench/
  config.py            # Pydantic config models + .env loading
  runner.py            # Async benchmark runner with checkpoint/resume
  metrics.py           # Latency percentiles, cost, effective-cost-per-million
  models.py            # RequestRecord, Timing, RateStats dataclasses
  checkpoint.py        # JSONL checkpoint for resume
  providers/
    client.py          # Streaming OpenAI-compatible client with timing captures
    retry.py           # Exponential backoff for 429/5xx
    ratelimit.py       # Per-provider semaphore + leaky bucket
  datasets/            # MMLU, GSM8K, ARC, HumanEval loaders
  scoring/             # Answer extraction + parity comparison
  reporting/
    json_out.py        # JSON report with precomputed aggregates
    markdown.py        # Markdown report with comparison tables
    cost_model.py      # Cross-provider cost comparison (metered + subscription)
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```
