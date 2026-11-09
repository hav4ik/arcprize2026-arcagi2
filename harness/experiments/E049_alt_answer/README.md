# E049: a free second guess inside attempt 0 (`--alt-answer`)

2026-10-26. On Kaggle (4xL4, E128) only ~2–19% of tasks finish an attempt 1 (E047 `kaggle_sim.py`), but the eval gap
between pass@1 and pass@2 of two fresh Flash-1017 attempts is ~8–11 pp. Idea: let attempt 0 record a second guess at
no extra rollout, scored as attempt_2.

Harness option (opt-in, `evals/run_deepseek_tools.py`; defaults and the pinned Kaggle prompt unchanged; tests pass):
`--alt-answer {passive,active}` appends a note to the prompt; after the attempt the harness reads the session variable
`alt_answer` (`submit_answer(session, "alt_answer", multi)`, no model tokens) and keeps it only if it is a valid grid
(list of grids for multi-test) that differs from the main answer. Records get `alt_pred` / `alt_correct` (per pair).
- `passive`: "Optional second guess: if you meet a different rule that also reproduces every training pair but gives a
  different test output, set `alt_answer` ... do not spend extra effort looking for one."
- `active`: "You get two tries per test output. Before submitting, briefly ask whether a different rule also reproduces
  every training pair but gives a different test output; if so, set `alt_answer` ..."

`scripts/run.sh <passive|active>`: all 120 eval tasks, the Kaggle candidate's flags (as E046 `e046p_note_all`:
sub80s40 + header, multi-test + multi-budget 0.5, fallback-solve, train-check, turns-note 5), Flash-1017 via OpenRouter
DeepInfra, 2 fresh samples (attempt 0 and 1 are independent). Result files keep the original tag `e048or_<mode>`
(the experiment was renamed from E048 after an ID clash). `scripts/analyze.py` compares.

## Results (120 tasks, 2 samples per mode, 0 errors after retrying provider timeouts)

| run | samples | pass@1 (main) | free pass@2 (main or alt) | alt set (pairs) | alt rescues (main wrong, alt right) | out tok / task-attempt |
|---|---|---|---|---|---|---|
| fresh, no note (E046 a0/a1 DeepSeek API + E047 OpenRouter ctrl a1–a3) | 5 | 76.1% (70.8–80.0) | = pass@1 | – | – | 29.4k |
| `passive` | 2 | 72.6% (71.9 / 73.2) | 73.2% | 6–10% | 1 + 1 | 30.6k |
| `active` | 2 | 74.1% (72.8 / 75.4) | 74.7% | 8–9% | 1 + 1 | 30.2k |

- The model rarely records an alternative, and when it does the alternative is usually wrong: 4 rescues in 4 runs,
  +0.6 pp, against a pass@1 that is 2–3 pp lower than the fresh baseline (within noise: sd across fresh samples ~3 pp,
  but no sign of a gain). Tokens unchanged.
- The 8–11 pp pass@2 gap of two fresh attempts comes from independent rollouts that end in different rules; one
  rollout does not surface its own competing rule, even when asked to look.
- Not adopted. Kaggle impact <= ~0.5 point, so no simulator run.
