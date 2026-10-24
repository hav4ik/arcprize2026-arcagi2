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

## Results
(pending)
