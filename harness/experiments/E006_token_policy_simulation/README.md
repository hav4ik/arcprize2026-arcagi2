# E006: Token-saving policies simulated on existing results (no API cost)

## Question
Can simple, model-agnostic policies cut DeepSeek V4.1-Flash output tokens without losing accuracy? Candidates are output-token caps and skipping the second attempt.

## Setup
- Data: E001d records (`evals/results/arcprize_deepseek-flash_max.jsonl`): ARC Prize prompt, max effort, 2 attempts × 167 test pairs. 317 attempts succeeded; 17 were lost to connection errors.
- Each policy is replayed on the recorded attempts.
- A capped attempt counts as wrong if it used more tokens than the cap. That assumes truncation means no answer, which matches what we saw.

## Results

### Per-call output-token cap (ARC Prize pass@2 over 120 tasks)
| Cap | pass@2 | Δ | Total output tokens | Δ |
|---|---|---|---|---|
| none | 78.1% | — | 16.95M (53.5k/call) | — |
| 120k | 77.6% | −0.4 pp | 16.89M | −0% |
| 100k | 76.4% | −1.7 pp | 16.56M | −2% |
| 80k | 73.1% | −5.0 pp | 15.60M | −8% |
| 60k | 65.1% | −12.9 pp | 14.04M | −17% |
| 40k | 41.4% | −36.7 pp | 11.03M | −35% |

### Skip the 2nd attempt when the 1st was short (152 pairs with both attempts)
| Policy | Pairs solved | Δ | Tokens | Δ |
|---|---|---|---|---|
| always 2 attempts | 79.6% | — | 15.90M | — |
| skip if 1st < 20k | 78.9% | −0.7 pp | 15.73M | −1% |
| skip if 1st < 30k | 77.6% | −2.0 pp | 14.92M | −6% |
| skip if 1st < 40k | 75.0% | −4.6 pp | 13.76M | −13% |
| always 1 attempt | 65.8% | −13.8 pp | 7.81M | −51% |

### Signals
**Attempt-1 accuracy by its token count:**

| Attempt-1 tokens | n | Correct |
|---|---|---|
| 0–30k | 37 | 89% |
| 30–45k | 35 | 69% |
| 45–60k | 31 | 77% |
| 60–80k | 23 | 61% |
| >80k | 26 | 19% |

**Agreement between attempts:** the two attempts give identical answers on 77/152 pairs (51%), and those answers are correct 92.2% of the time. When they disagree, either attempt is correct 66.7% of the time.

## Conclusions
- **Caps and length-based skipping can't cut tokens without losing accuracy.** Even the mildest useful setting costs about 1–2 pp for ≤6% fewer tokens. Many correct answers need long reasoning: 35% of correct attempts used more than 45k tokens.
- **Very long attempts are mostly wrong** (19% correct above 80k). Stopping them early would not save accuracy, though, because the second attempt is what recovers those tasks. The saving has to come from making each attempt cheaper.
- **Agreement is a strong confidence signal** (92% precision). It needs two answers, but they could come from a cheaper second pass, such as a lower-effort attempt or a re-check.
- **Implication:** tool-free policies are not where the token savings are. The E003 trace analysis points to the mechanical work (25–45% of tokens on building and counting grids by hand) as the main waste. Removing it needs code execution, i.e. the planned tool-use harness.
