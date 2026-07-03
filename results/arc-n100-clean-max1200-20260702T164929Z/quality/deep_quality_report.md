# Additional Quality Analysis: ARC Correctness and Failure Modes

This is a deeper read of the existing merged ARC n=100 data. It joins raw provider outputs with ARC gold labels and separates answer correctness from operational / generation failures.

## Headline

- Among responses that produced a parseable answer, provider answered-only accuracy stayed high on this ARC n=100 sample, but parsed wrong answers still occurred.
- Empty final answers after spending the whole 1200-token completion budget on reasoning are a separate failure mode from wrong parsed answers.
- Therefore the current benchmark is measuring a mix of: (1) ARC capability, (2) answer-format reliability, and (3) whether the 1200-token completion budget is enough when reasoning is enabled.
- Exact single-letter compliance varies by provider; see the provider-level table and empty-content table below for request-level details.

## Provider-level correctness

| Provider | Requests | Parseable correct | Empty content | Strict success rate | Answered-only accuracy | Exact single-letter outputs | Length stops | Median E2E non-empty | Median E2E empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| umans | 200 | 187 | 5 | 93.5% | 96.9% | 193 | 5 | 12,824 ms | 36,961 ms |
| zai | 200 | 190 | 3 | 95.0% | 97.4% | 192 | 3 | 5,676 ms | 26,509 ms |

## Empty-content failures

| Provider | Item | Pass | Gold | Stop | Output tokens | Reasoning tokens | E2E | Question short |
|---|---|---|---|---|---:|---:|---:|---|
| zai | arc-0070 | latency | A | length | 1200 | 1197 | 26,509 ms | Garden plants on Earth require four resources to stay alive: soil, air, water, and sunlight. Ho... |
| zai | arc-0072 | parity | B | length | 1200 | 1199 | 28,971 ms | Some scientists predict that as greenhouse gases change the climate of Earth, the number and st... |
| zai | arc-0072 | latency | B | length | 1200 | 1200 | 25,268 ms | Some scientists predict that as greenhouse gases change the climate of Earth, the number and st... |
| umans | arc-0029 | latency | B | length | 1200 | 1203 | 36,961 ms | Which statement best explains why a tree branch floats on water?... |
| umans | arc-0035 | parity | A | length | 1200 | 1200 | 44,983 ms | A scientist maps a long region in which earthquakes originate and determines this region is a t... |
| umans | arc-0064 | latency | D | length | 1200 | 1204 | 36,708 ms | Dominic is observing a volvox colony and a paramecium under a microscope. He makes a note in hi... |
| umans | arc-0070 | parity | A | length | 1200 | 1203 | 57,652 ms | Garden plants on Earth require four resources to stay alive: soil, air, water, and sunlight. Ho... |
| umans | arc-0072 | parity | B | length | 1200 | 0 | 16,830 ms | Some scientists predict that as greenhouse gases change the climate of Earth, the number and st... |

## Item-level outcome taxonomy

| Outcome | Count | Items |
|---|---:|---|
| Both providers 2/2 correct | 92 | arc-0000, arc-0001, arc-0002, arc-0003, arc-0004, arc-0005, arc-0006, arc-0007, arc-0008, arc-0009, arc-0010, arc-0011, arc-0012, arc-0013, arc-0014, arc-0015, arc-0016, arc-0017, arc-0018, arc-0019, arc-0020, arc-0021, arc-0022, arc-0023, arc-0024, arc-0025, arc-0026, arc-0027, arc-0028, arc-0030, arc-0031, arc-0032, arc-0033, arc-0034, arc-0036, arc-0037, arc-0038, arc-0039, arc-0040, arc-0041, arc-0042, arc-0043, arc-0045, arc-0046, arc-0047, arc-0048, arc-0049, arc-0050, arc-0051, arc-0052, arc-0053, arc-0055, arc-0056, arc-0057, arc-0058, arc-0059, arc-0060, arc-0061, arc-0062, arc-0063, arc-0065, arc-0066, arc-0068, arc-0069, arc-0071, arc-0073, arc-0074, arc-0075, arc-0076, arc-0077, arc-0078, arc-0079, arc-0080, arc-0081, arc-0082, arc-0083, arc-0084, arc-0085, arc-0086, arc-0087, arc-0088, arc-0089, arc-0090, arc-0091, arc-0092, arc-0093, arc-0094, arc-0095, arc-0096, arc-0097, arc-0098, arc-0099 |
| At least one provider unstable: one pass correct, one pass empty | 3 | arc-0029, arc-0035, arc-0064 |
| Both providers have full-empty item or severe cap issue | 2 | arc-0044, arc-0072 |
| Other mixed outcome | 3 | arc-0054, arc-0067, arc-0070 |

## What else we can analyze with existing data

- Strict task success: answer emitted and correct, not just HTTP 200.
- Format adherence / answerability: empty output, non-letter output, multiple-letter output, and exact single-letter compliance.
- Reasoning budget failures: stop_reason=length, output_tokens=max_tokens, reasoning_tokens≈max_tokens.
- Pass-to-pass stability at temperature 0: parity vs latency pass per item/provider.
- Latency-quality coupling: empty/capped outputs are much slower, especially on umans.
- Provider-specific hard items: items where one provider emits an answer and the other exhausts reasoning budget.
- Prompt/item sensitivity: correlate failure with question length, answer label, and scientific subtopic (needs metadata/classification).
- Benchmark design sensitivity: rerun only failed/empty items with larger max_tokens or reasoning_effort=none to separate model knowledge from harness settings.
- Confidence in conclusions: bootstrap CIs and scale further if tighter intervals are needed.
- Data schema improvement: store gold labels, choices, and scoring fields inside results.json so quality analysis is reproducible without reloading HF.

