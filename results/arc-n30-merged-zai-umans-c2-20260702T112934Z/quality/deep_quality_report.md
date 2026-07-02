# Additional Quality Analysis: ARC Correctness and Failure Modes

This is a deeper read of the existing merged ARC n=30 data. It joins raw provider outputs with ARC gold labels and separates answer correctness from operational / generation failures.

## Headline

- Among responses that produced a parseable answer, **both providers were 100% correct** on this small ARC sample.
- The observed quality gap is not wrong answers. It is **empty final answers after spending the whole 300-token completion budget on reasoning**.
- Therefore the current benchmark is measuring a mix of: (1) ARC capability, (2) answer-format reliability, and (3) whether `max_tokens=300` is enough when reasoning is enabled.
- One Z.ai response was `Answer: B` rather than exactly `B`; it is correct under parser scoring but fails the strict 'only the letter' instruction.

## Provider-level correctness

| Provider | Requests | Parseable correct | Empty content | Strict success rate | Answered-only accuracy | Exact single-letter outputs | Length stops | Median E2E non-empty | Median E2E empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| umans | 60 | 48 | 12 | 80.0% | 100.0% | 48 | 12 | 4,817 ms | 17,084 ms |
| zai | 60 | 52 | 8 | 86.7% | 100.0% | 51 | 8 | 4,652 ms | 7,774 ms |

## Empty-content failures

| Provider | Item | Pass | Gold | Stop | Output tokens | Reasoning tokens | E2E | Question short |
|---|---|---|---|---|---:|---:|---:|---|
| zai | arc-0004 | parity | D | length | 300 | 299 | 6,744 ms | An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total d... |
| zai | arc-0004 | latency | D | length | 300 | 299 | 6,339 ms | An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total d... |
| zai | arc-0005 | parity | B | length | 300 | 300 | 6,312 ms | Devil facial tumor disease (DFTD) is a disease that is decimating the population of Tasmanian d... |
| zai | arc-0011 | latency | A | length | 300 | 296 | 7,232 ms | Which statement best describes the effect of the Sun on the oceans?... |
| zai | arc-0028 | parity | C | length | 300 | 299 | 8,891 ms | Tiny organisms called plankton live in oceans. Some plankton can take energy from the Sun and t... |
| zai | arc-0028 | latency | C | length | 300 | 296 | 8,317 ms | Tiny organisms called plankton live in oceans. Some plankton can take energy from the Sun and t... |
| zai | arc-0029 | parity | B | length | 300 | 298 | 9,201 ms | Which statement best explains why a tree branch floats on water?... |
| zai | arc-0029 | latency | B | length | 300 | 300 | 9,171 ms | Which statement best explains why a tree branch floats on water?... |
| umans | arc-0004 | parity | D | length | 300 | 300 | 27,610 ms | An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total d... |
| umans | arc-0004 | latency | D | length | 300 | 304 | 26,369 ms | An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total d... |
| umans | arc-0005 | parity | B | length | 300 | 302 | 79,629 ms | Devil facial tumor disease (DFTD) is a disease that is decimating the population of Tasmanian d... |
| umans | arc-0005 | latency | B | length | 300 | 300 | 83,804 ms | Devil facial tumor disease (DFTD) is a disease that is decimating the population of Tasmanian d... |
| umans | arc-0010 | parity | B | length | 300 | 302 | 5,946 ms | According to cell classification, prokaryotic cells are separated from eukaryotic cells. Which ... |
| umans | arc-0011 | parity | A | length | 300 | 302 | 52,375 ms | Which statement best describes the effect of the Sun on the oceans?... |
| umans | arc-0011 | latency | A | length | 300 | 300 | 47,071 ms | Which statement best describes the effect of the Sun on the oceans?... |
| umans | arc-0019 | parity | C | length | 300 | 303 | 6,160 ms | It was once thought that living organisms could come from non-living matter. For example, peopl... |
| umans | arc-0020 | parity | B | length | 300 | 302 | 6,106 ms | Cells take in food for energy. The part of the cell that aids in digestion of the food is the l... |
| umans | arc-0028 | parity | C | length | 300 | 304 | 5,418 ms | Tiny organisms called plankton live in oceans. Some plankton can take energy from the Sun and t... |
| umans | arc-0028 | latency | C | length | 300 | 301 | 5,086 ms | Tiny organisms called plankton live in oceans. Some plankton can take energy from the Sun and t... |
| umans | arc-0029 | latency | B | length | 300 | 300 | 7,800 ms | Which statement best explains why a tree branch floats on water?... |

## Item-level outcome taxonomy

| Outcome | Count | Items |
|---|---:|---|
| Both providers 2/2 correct | 22 | arc-0000, arc-0001, arc-0002, arc-0003, arc-0006, arc-0007, arc-0008, arc-0009, arc-0012, arc-0013, arc-0014, arc-0015, arc-0016, arc-0017, arc-0018, arc-0021, arc-0022, arc-0023, arc-0024, arc-0025, arc-0026, arc-0027 |
| Both providers have full-empty item or severe cap issue | 2 | arc-0004, arc-0028 |
| At least one provider unstable: one pass correct, one pass empty | 6 | arc-0005, arc-0010, arc-0011, arc-0019, arc-0020, arc-0029 |

## What else we can analyze with existing data

- Strict task success: answer emitted and correct, not just HTTP 200.
- Format adherence / answerability: empty output, non-letter output, multiple-letter output, and exact single-letter compliance.
- Reasoning budget failures: stop_reason=length, output_tokens=max_tokens, reasoning_tokens≈max_tokens.
- Pass-to-pass stability at temperature 0: parity vs latency pass per item/provider.
- Latency-quality coupling: empty/capped outputs are much slower, especially on umans.
- Provider-specific hard items: items where one provider emits an answer and the other exhausts reasoning budget.
- Prompt/item sensitivity: correlate failure with question length, answer label, and scientific subtopic (needs metadata/classification).
- Benchmark design sensitivity: rerun only failed/empty items with larger max_tokens or reasoning_effort=none to separate model knowledge from harness settings.
- Confidence in conclusions: bootstrap CIs on n=30, then scale to n>=100 because this sample is small and label distribution is imbalanced.
- Data schema improvement: store gold labels, choices, and scoring fields inside results.json so quality analysis is reproducible without reloading HF.

