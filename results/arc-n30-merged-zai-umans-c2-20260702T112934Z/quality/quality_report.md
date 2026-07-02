# GLMsBench ARC n=30 Quality Analysis

This analysis joins provider outputs to ARC-Challenge gold labels through the existing `ARCLoader`. It covers correctness, parseability, item-level stability, and quality/latency coupling.

## Provider accuracy

| Provider | Correct / requests | Strict success | Answered-only accuracy | Empty content | Exact single-letter | Length stops | Stable correct items |
|---|---:|---:|---:|---:|---:|---:|---:|
| umans | 48 / 60 | 80.0% | 100.0% | 12 | 48 | 12 | 22 / 30 |
| zai | 52 / 60 | 86.7% | 100.0% | 8 | 51 | 8 | 25 / 30 |

## Provider latency by quality bucket

| Provider | Median E2E non-empty | Median E2E empty | Median E2E correct | Median E2E unknown |
|---|---:|---:|---:|---:|
| umans | 4,817 ms | 17,084 ms | 4,817 ms | 17,084 ms |
| zai | 4,652 ms | 7,774 ms | 4,652 ms | 7,774 ms |

## Item patterns

- Items where both providers were correct on both passes: **22/30**
- Items with any empty/non-parseable output: **8/30** - arc-0004, arc-0005, arc-0010, arc-0011, arc-0019, arc-0020, arc-0028, arc-0029

## Gold answer distribution

| Label | Count |
|---|---:|
| A | 5 |
| B | 11 |
| C | 11 |
| D | 3 |

## Analysis gaps unlocked by current data

- Add gold labels to normalization output so accuracy/correctness is first-class, not a separate join.
- Report request-level accuracy, item-level accuracy, answered-only accuracy, and unknown parse rate separately.
- Track pass-to-pass stability: parity and latency passes can disagree even at temperature 0.
- Separate benchmark capability from provider reliability: correctness only among completed requests hides failed-request impact.
- Analyze hard items: list items missed by both providers, missed by only one provider, and unknown/non-letter outputs.
- Measure latency-quality coupling: compare latency/token counts for correct vs incorrect/unknown outputs.
- Scale n beyond 30 because the current sample has high variance and answer-label imbalance.
- Store dataset metadata (question, choices, gold, category) in raw results for reproducible audits without reloading HF.

## Artifacts

- `quality/quality_summary.json`
- `quality/item_correctness.csv`
- `quality/deep_quality_report.md`
