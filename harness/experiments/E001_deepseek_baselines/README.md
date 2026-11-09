# E001: DeepSeek baselines with a plain prompt

## Hypothesis
DeepSeek V4-class models solve a large fraction of ARC-AGI-2 with a single plain prompt. ARC Prize reports 61.4% semi-private for V4 Flash 0731 (max effort). These baselines are the reference point for harnesses (E002) and for distillation targets.

## Setup
- **Eval set:** the ARC-AGI-2 part of NVIDIA's `Nemotron-RL-ARC-AGI-v1` validation split.
  - 120 problems, one test input each. Some are D4/colour/scale/pad-transformed, and each has at most ~5 train pairs.
  - Local copy: `data/nemotron_rl/transductive_validation.jsonl`.
- **Prompt:** NVIDIA's transductive prompt as-is (system + user; grids as digit rows; answer in `\boxed{}`).
- **Scoring:** exact match against NVIDIA's `expected_output`. That field matches the GitHub ground truth, which differs from our Kaggle solutions file on 4 tasks.
- **API settings:** DeepSeek API, thinking mode enabled, streaming, `max_tokens=200000`, 24 concurrent requests. Temperature is not settable in thinking mode.
- **Command:**
  ```bash
  .venv/bin/python evals/run_deepseek_arc2.py --model <deepseek-flash|deepseek-v4-pro> --effort <high|max>
  ```

| Run | Model | Effort | Results file |
|---|---|---|---|
| E001a | `deepseek-flash` (V4.1-Flash) | max | `evals/results/arc2_deepseek-flash_max.jsonl` |
| E001b | `deepseek-flash` (V4.1-Flash) | high | `evals/results/arc2_deepseek-flash_high.jsonl` |
| E001c | `deepseek-v4-pro` | max | `evals/results/arc2_deepseek-v4-pro_max.jsonl` |

For E001a and E001b, a second sample per problem (`--samples 2`) gives pass@2.

### E001d: replica of ARC Prize's setup
Purpose: a number directly comparable to ARC Prize's public-eval score (63.9% for V4 Flash 0731 at max effort).

- **Tasks:** the official 120 tasks / 167 test pairs from `github.com/arcprize/ARC-AGI-2`, cloned to `data/arc-agi-2-github/`.
- **Prompt:** ARC Prize's prompt verbatim as a single user message, with grids as `json.dumps`. Verified byte-identical to the prompt in ARC Prize's recorded DeepSeek V3.2 run.
- **Parsing:** ARC Prize's parser (`\boxed{json}`, else the last list of lists).
- **Scoring:** 2 attempts per test pair; a task's score is the fraction of its pairs solved, averaged over the 120 tasks.
- **Command:**
  ```bash
  cd evals && ../.venv/bin/python run_deepseek_arcprize.py --model deepseek-flash --effort max
  ```
- **Results file:** `evals/results/arcprize_deepseek-flash_max.jsonl`.
- **Unknown:** ARC Prize's provider, max_tokens and temperature for V4 are unpublished. We use DeepSeek API defaults with `max_tokens=200000`.

## Results (2026-10-01)

**NVIDIA's ARC-AGI-2 set** (120 problems, one test input each, NVIDIA prompt):

| Run | pass@1 | pass@2 | Output tokens/call | Cost |
|---|---|---|---|---|
| E001a Flash (V4.1), max | 52.9% (mean of 2 samples) | 60.8% | 81k | ~$12 |
| E001b Flash (V4.1), high | 40.8% | 56.7% | 72k | ~$10 |
| E001c V4-Pro, max | 40.0% (1 sample) | — | 75k | ~$18 |

**E001d, official public eval with ARC Prize's setup** (120 tasks / 167 pairs):
- pass@2: **78.1%**; pass@1 (attempt 1 only): 63.1%. Output: 53k tokens per call; cost ~$10.
- Both are lower bounds: 17 of 334 attempts failed with `APIConnectionError` on long streams and count as wrong.
- Reference: ARC Prize reports V4 Flash 0731 at max effort as 63.9% public / 61.4% semi-private.

**Prompt effect.** On the 61 problems that are identical in both sets (same train pairs, same test input), Flash max scored:

| Prompt | pass@1 | pass@2 |
|---|---|---|
| NVIDIA prompt (digit rows, `\boxed{}`) | 42.6% | 50.8% |
| ARC Prize prompt (JSON grids) | 53.3% | 72.1% |

Costs are at off-peak list prices (Flash $0.60/M output tokens, Pro $1.98/M). The National Day holiday makes the whole day off-peak. Total spend was ~$55, including ~$2–4 of calls that were in flight when the runs were restarted with more workers.

### Run log and cost accounting

The DeepSeek dashboard shows 1,032 calls (887 Flash, 145 Pro) and ~$35 Flash + ~$20 Pro. That reconciles exactly with the table below.

| # | Run | Model | Calls | Output tokens | Notes |
|---|---|---|---|---|---|
| 1 | API smoke test | Flash | 1 | ~0 | |
| 2 | E001a 1st sample (incl. 4-problem pilot) | Flash | 120 | 9.97M | |
| 3 | E001b 1st sample | Flash | 120 | ~8.6M | |
| 4 | E001c pilot (interrupted) | Pro | 3 | ~0.1M | 2 recorded, 1 cut off |
| 5 | E001c | Pro | 118 | ~8.9M | |
| 6 | E001d (2 attempts × 167 pairs) | Flash | 334 | 16.95M | 17 `APIConnectionError`s; partial output billed, not recorded |
| 7 | E001a 2nd sample | Flash | 120 | ~9.5M | |
| 8 | E001b 2nd sample | Flash | 120 | ~8.7M | |
| 9 | Calls in flight when runs were restarted with more workers | 72 Flash + 24 Pro | 96 | not recorded | Billed and wasted (~$5) |

Recorded calls: 934. Unrecorded but billed: 98 (rows 4, 9, and the cut-off part of row 6).

**Per-call cost (Flash, off-peak $0.60/M output tokens; input is ~$0.001/call):**

| Setup | Output tokens/call | $/call |
|---|---|---|
| NVIDIA prompt, max | 81k | ~0.049 |
| NVIDIA prompt, high | 72k | ~0.043 |
| ARC Prize prompt, max | 53k | ~0.033 |
| Overall ($35 / 887 billed calls) | — | 0.039 |

- **Per task:** ARC Prize-style pass@2 (E001d) is ~$0.09/task. ARC Prize reports $0.042/task for V4 Flash 0731; their token cap and provider are unknown.
- **Rule of thumb:** calls × ~60k tokens × $0.60/M ≈ $0.035–0.05 per call off-peak. Double that at peak.

**Lessons:**
- Estimate spend from tokens × list price. The account balance lags behind usage, and inferring the rate from it under-estimated cost by ~2×. The account overdrew to −$4.92.
- Pilot runs under-estimate tokens per call, because short calls finish first.
- Restarting a run re-bills every call that is in flight. Choose the concurrency at launch (DeepSeek's limits are 2500 concurrent for Flash, 500 for Pro).

## Conclusions
- **Prompt and format matter a lot.** On identical problems, the ARC Prize prompt beats NVIDIA's by about 11 points pass@1 and 21 points pass@2, while using fewer tokens.
- **V4.1-Flash at max effort beats ARC Prize's public number for V4 Flash 0731** (78.1% vs 63.9% pass@2). Possible causes:
  - V4.1 is genuinely better.
  - Our settings differ: `max_tokens=200000`, the native DeepSeek API.
  - Contamination: the public eval has been out since 2025. We can't tell these apart without semi-private access.
- **V4-Pro is worse than Flash on this task.** At max effort it scored 40.0% vs Flash's 52.9% pass@1, at about 1.5× the cost per call and 3× the latency. Flash is the right model for harness work (E002).
- **Max effort beats high** by about 12 points pass@1 and 4 points pass@2. High benefits more from a second sample.
- **Failure modes are rare.** A few empty final answers (all reasoning, no answer) and one call truncated at 200k tokens. Retrying on empty answers would recover a little.

## Next
- Per-difficulty breakdown, failure analysis (wrong vs unparsed vs truncated), and cost per correct answer.
- Optionally rerun with ARC Prize's official prompt (`notes/prompts/`), to see how sensitive the score is to the prompt.
