# GLMsBench — GLM-5.2 Provider Benchmark

Compares GLM-5.2 performance between Z.ai and umans across MMLU, GSM8K,
ARC-Challenge, and HumanEval. Measures latency (TTFT, inter-token, e2e),
output parity (temp=0), cost, and rate-limit behavior. Reports JSON + Markdown.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"         # macOS/Linux
cp .env.example .env   # then fill in ZAI_API_KEY / UMANS_API_KEY
```

API keys are loaded from a `.env` file in the project root (auto-loaded by
`load_config`, via `python-dotenv`) — `.env` is gitignored, never commit it.
Real shell environment variables (`export`/`$env:`) still work and take
precedence over `.env` if both are set.

## Run

```bash
.venv/Scripts/python.exe run.py --config harness/config.example.yaml --out results/
```

Flags: `--suite mmlu,gsm8k` (subset), `--out results/`, `--resume`.

## Smoke test (1-item, both providers, ~minimal spend)

```bash
.venv/Scripts/python.exe scripts/smoke_test.py --config harness/config.example.yaml
```

## What it reports

- **Accuracy**: per-provider sample counts per benchmark (per-item scoring via
  `glmsbench.scoring`).
- **Parity**: cross-provider answer disagreement rate (temp=0) — the headline
  number for "do Z.ai and umans behave identically?".
- **Latency**: clean vs effective p50/p95 for TTFT / inter-token / e2e.
  *Clean* excludes rate-limited requests; *effective* includes backoff.
- **Cost**: total per provider (input + output tokens × pricing).
- **Rate limits**: throttles (429s), backoff time, avg retries per provider.

## Architecture

Single `glmsbench` package of single-purpose modules: provider-agnostic
streaming client, dataset loaders, pure scoring functions, metrics, a runner
with checkpoint/resume, and JSON + Markdown reporters.

See `docs/superpowers/specs/2026-06-26-glm-5.2-provider-benchmark-design.md`
for the full design and `docs/superpowers/plans/2026-06-26-glm-5.2-provider-benchmark.md`
for the implementation plan.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```
