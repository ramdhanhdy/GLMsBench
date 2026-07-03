# GLMsBench ARC n=100 Quality Analysis

This analysis joins provider outputs to ARC-Challenge gold labels through the existing `ARCLoader`. It covers correctness, parseability, item-level stability, and quality/latency coupling.

## Provider accuracy

| Provider | Correct / requests | Strict success | Answered-only accuracy | Empty content | Exact single-letter | Length stops | Stable correct items |
|---|---:|---:|---:|---:|---:|---:|---:|
| umans | 187 / 200 | 93.5% | 96.9% | 5 | 193 | 5 | 92 / 100 |
| zai | 190 / 200 | 95.0% | 97.4% | 3 | 192 | 3 | 95 / 100 |

## Provider latency by quality bucket

| Provider | Median E2E non-empty | Median E2E empty | Median E2E correct | Median E2E unknown |
|---|---:|---:|---:|---:|
| umans | 12,824 ms | 36,961 ms | 11,550 ms | 36,708 ms |
| zai | 5,676 ms | 26,509 ms | 5,624 ms | 25,268 ms |

## Item patterns

- Items where both providers were correct on both passes: **92/100**
- Items with any empty/non-parseable output: **6/100** - arc-0029, arc-0035, arc-0044, arc-0064, arc-0070, arc-0072

## Gold answer distribution

| Label | Count |
|---|---:|
| 2 | 1 |
| A | 23 |
| B | 26 |
| C | 33 |
| D | 17 |

## Analysis gaps unlocked by current data

- Add gold labels to normalization output so accuracy/correctness is first-class, not a separate join.
- Report request-level accuracy, item-level accuracy, answered-only accuracy, and unknown parse rate separately.
- Track pass-to-pass stability: parity and latency passes can disagree even at temperature 0.
- Separate benchmark capability from provider reliability: correctness only among completed requests hides failed-request impact.
- Analyze hard items: list items missed by both providers, missed by only one provider, and unknown/non-letter outputs.
- Measure latency-quality coupling: compare latency/token counts for correct vs incorrect/unknown outputs.
- Scale beyond the current sample when tighter confidence intervals are needed.
- Store dataset metadata (question, choices, gold, category) in raw results for reproducible audits without reloading HF.

## Artifacts

- `quality/quality_summary.json`
- `quality/item_correctness.csv`
- `quality/deep_quality_report.md`
