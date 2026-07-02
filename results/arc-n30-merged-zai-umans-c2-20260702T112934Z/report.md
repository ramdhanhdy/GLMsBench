# GLMsBench ARC n=30 Provider Decision Report

## Executive summary

This run is primarily a **reliability failure story**, not a clean latency race. Z.ai completed 60/60 requests; umans completed 60/60 and then entered a throttle/suspension failure mode. Latency and token charts below are therefore explicitly limited to successful requests.

This report compares Z.ai and umans serving GLM-5.2 on a 30-item ARC sample. It is a small but chartable run designed to validate the benchmark-to-report workflow before scaling.

## Run facts

| Field | Value |
|---|---:|
| Requests | 120 |
| Providers | umans, zai |
| Suites | arc |

## Provider summary

| Provider | OK / Total | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 | Output tok p50 | Throttles |
|---|---:|---:|---:|---:|---:|---:|---:|
| umans | 60 / 60 | 4816.5 | 28107.2 | 5719.3 | 55143.9 | 177 | 0 |
| zai | 60 / 60 | 4652.0 | 8160.5 | 4860.0 | 8905.3 | 154 | 0 |

## Parity

| Suite | Compared | Canonical disagreements | Unknown | Disagreement rate | Raw text agreement |
|---|---:|---:|---:|---:|---:|
| arc | 30 | 0 | 8 | 0.0% | 80.0% |

## Charts

### Provider scorecard

![Provider scorecard](charts/provider_scorecard_table.png)

*Head-to-head scorecard. Latency and token stats are computed on successful requests only; no composite winner score is used.*

### Request outcome by provider

![Request outcome by provider](charts/request_outcomes.png)

*Request outcomes per provider, n=60 each. Z.ai returned 60/60 OK; umans returned 14/60 OK and 46 errors.*

### Throttling and backoff diagnostics

![Throttling and backoff diagnostics](charts/throttling_diagnostics.png)

*Provider lifecycle diagnostics. umans saw 16 throttled events and 24.3s cumulative backoff; Z.ai saw none.*

### End-to-end latency distribution

![End-to-end latency distribution](charts/e2e_latency_success_only.png)

*Successful requests only. umans n=14, median 6773 ms; Z.ai n=60, median 4860 ms.*

### Time to first token

![Time to first token](charts/ttft_success_only.png)

*Measured successful requests only. umans n=11, median 6307 ms; Z.ai n=52, median 4652 ms.*

### Output token distribution

![Output token distribution](charts/output_tokens_success_only.png)

*Successful responses only. umans median 212.5 tokens; Z.ai median 154 tokens. p95=300 for both, likely from a cap.*

### Paired parity subset

![Paired parity subset](charts/paired_parity_subset.png)

*Paired subset only. 7 comparable items, 0 canonical disagreements, 2 unknown extractions. Not a full-suite agreement rate.*
