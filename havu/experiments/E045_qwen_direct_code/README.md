# E045: Qwen3.8-27B writes solver code directly (no thinking), sample many, keep train-passing programs

Idea (user, 2026-10-24): skip thinking and the tool loop. Ask Qwen3.8-27B in non-thinking mode for a
`solve(grid)` program, sample 32 per task, keep only programs that reproduce **every** training pair, and vote
their test outputs into pass@2.

## Arms (32 samples per task each)
| Arm | Prompt | Checked on |
|---|---|---|
| `code` | all training pairs + test inputs; reply = one ```python block | all training pairs |
| `rulecode` | same, reply = `<rule>…</rule><code>…</code>` (state the rule, then implement it) | all training pairs |
| `rulecode_loo` | like `rulecode`, but one training pair is left out of the prompt (rotating; 32 samples split over the pairs) | **all** training pairs, so the held-out pair is a real generalisation test |

## Setup
- Data: Kaggle `arc-agi_evaluation_challenges.json` (120 tasks, 172 test inputs), solutions only on the host.
- Model: `Qwen/Qwen3.8-27B-FP8` (block FP8; Marlin weight-only on SM86), vLLM 0.32, TP4 on Vast 4×RTX 3090
  (57428605), prefix caching on, `--language-model-only`, max-model-len 32k.
  `chat_template_kwargs: {enable_thinking: false}`.
- Sampling: temperature 0.8, top_p 0.95, top_k 20, no presence penalty, max 6000 tokens. Qwen's non-thinking
  recommendation includes presence_penalty 1.5, which penalises re-using identifiers and digits in code, so it is off.
  One request per prompt with `n` samples (the prompt is prefilled once).
- Checking (`validate.py`): each unique program runs once in the local Docker sandbox (no network, no host mounts),
  fresh process, 5 s per grid, 2 GB memory. Training outputs are compared in the sandbox; test outputs return to
  the host.
- Scoring (`score.py`): per test input, the outputs of train-passing samples are ranked by votes; top 2 distinct =
  attempt_1/2. Task score = mean over its test inputs, out of 120. Also pass@1 (top vote), oracle (any passing
  sample right), coverage (≥ 1 passing sample), and for `rulecode_loo` the share of programs that fit the shown
  pairs and also reproduce the held-out one.

Known weakness: a program that memorises the training outputs (a lookup table) passes the training check in the
`code`/`rulecode` arms. The LOO arm measures how often fitting the shown pairs generalises to a held-out pair.

## Files
`gen.py` (sampling, resumable), `validate.py` (sandbox checks, resumable), `score.py`, `scripts/` (box scripts).

## Results (2026-10-24): does not work; stopped after the pilot
- Pilot v1 (6000-token cap): without thinking the model reasons inside code comments ("Let's look at Ex 2 again...
  WAIT."); 12-26 of 32 samples per request hit the cap. Stopped.
- Pilot v2 (3000-token cap, "no reasoning inside the code"), 10 random eval tasks x 3 arms x 32 samples = 960:

  | Arm | parsed | hit cap | train-pass | tokens/sample |
  |---|---|---|---|---|
  | code | 55.0% | 33.8% | 0.3% | 2,065 |
  | rulecode | 93.4% | 41.2% | 0.0% | 1,793 |
  | rulecode_loo | 93.8% | 44.1% | 0.0% | 1,888 |

  **pass@2 0.00 / 10.** 775 unique programs: 670 got zero training pairs right, 38 syntax/indentation errors (cut-off
  code), the rest runtime errors. Only 1ae2feb7 had partial fits (5 programs; 1 passed all train pairs, wrong on test).
  LOO: 2 programs fit every shown pair, 0 also fit the held-out pair.
- Control (10 random ARC-AGI-2 *training* tasks, code arm, 8 samples): 1.00 / 10 (6.2% of samples train-pass), so the
  pipeline and scoring work; the model cannot write ARC-AGI-2 eval solvers in one shot without thinking.
- Throughput: ~1,000 tok/s on 4x3090 TP4 (FP8 Marlin), KV 721k tokens. The full 120-task run (~8 h) was not done.
- The box (Vast 57428605) was destroyed by the user on 2026-10-25; results files are in `results/`.
