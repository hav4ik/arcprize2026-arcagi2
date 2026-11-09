# E047: second attempt informed by the first attempt's answer (no thinking)

2026-10-26. Idea (user): attempt 2 should refine attempt 1 from its **final answer and rule (code)** only, not its
thinking: less KV than a transcript and less pull toward the same line of thought. Harness option (opt-in,
`evals/run_deepseek_tools.py`, defaults and the pinned Kaggle prompt unchanged; tests pass):
- `--prior-mode {verify,verifycode,differ} --prior-file F`: attempts >= 1 get attempt 0's answer grid(s) from F appended
  to the prompt, with "(its program reproduced every training pair)" when the host train check passed (a failed check
  is not shown: it misses many correct answers).
  - `verify`: check it; resubmit if confirmed (cheap), else fix and submit.
  - `verifycode`: the same + the program behind the answer, rebuilt from attempt 0's tool calls by
    `evals/answer_program.py`: the last definition of every function, import and assignment that the submitted expression
    (or the cell setting `answer`) needs; no exploration cells, output or thinking; median 1.4k chars, cap 4k (8/167
    truncated). Shown as text only, not loaded in the session.
  - `differ`: assume it is wrong, look for another rule (not run: see the Kaggle note).
- `hybrid`: `verifycode` when attempt 0 failed the train check, else a test-difference audit of the passing answer.
- `split`: `verifycode` when attempt 0 failed the train check, `verify` when it passed (the best offline mix).
- `scripts/run.sh <mode>` (PROVIDER=DeepInfra for OpenRouter, ATTEMPTS=3 for two informed samples) seeds the new file
  with attempt 0 of E046 `e046p_note_all` (DeepSeek API, the Kaggle candidate's flags), so only later attempts run.
  `scripts/compare.py` compares; `scripts/kaggle_sim.py` = E042 `multi_sim.py` with attempt 1 transformed.

## Results (Flash-1017 via OpenRouter DeepInfra, 120 eval tasks, attempts 1 and 2 = two samples, 0-1 errors)

pass@2 = (attempt 0, attempt k), task-weighted; "fixed" = a0 wrong -> ak right, "broke" = a0 right -> ak wrong (pairs).

| run | pass@2 a1 / a2 | pass@1 of ak | out tok per attempt | turns | same answer as a0 | fixed / broke | P(fix \| a0 wrong) | $ |
|---|---|---|---|---|---|---|---|---|
| fresh (E046's own a1, DeepSeek API) | 88.6% | 80.0% | 29.2k | 15.0 | 70% | 22 / 15 | 0.54 (22/41) | 4.96 |
| `ctrl` fresh | 83.6 / 83.6% | 75.3 / 79.0% | 29.0 / 29.6k | 14.8 | 74% | 27 / 20 | 0.35 (27/77) | 2.78 |
| `verify` | 86.1 / 84.0% | 82.4 / 77.4% | 27.8 / 26.6k | 13.5 | 79% | 30 / 19 | 0.37 (30/81) | 2.64 |
| **`verifycode`** | 83.6 / 78.6% | 82.8 / 78.6% | **16.2 / 16.8k (−44%)** | **10.1** | 88% | 18 / 2 | 0.22 (18/82) | 1.71 |
| `hybrid` | 82.6 / 82.4% | 78.4 / 80.7% | 21.6 / 22.5k | 11.1 | 85% | 24 / 10 | | 1.99 |
| `split` (3 samples) | 79.9 / 81.1 / 83.6% | 78.6 / 78.6 / 79.4% | 25.9 / 25.2 / 25.1k (−13%) | 13.0 | 83% | 29 / 13 | | 3.92 |
| `ctrl` 3rd sample (a3) | 84.4% | 70.8% | 29.1k | 14.5 | 67% | 15 / 21 | | 1.52 |

- `verify` (answer only) = a fresh attempt: the model re-solves; no token saving, pass@2 a tie.
- `verifycode` checks in 44% fewer tokens and 32% fewer turns and almost never breaks a right answer, but anchors: it
  fixes 0.6x as often as a fresh attempt, pass@2 −2.5 pp on average (two samples 5 pp apart).

**Kaggle estimate** (`kaggle_sim.py`: E128 run-3 records, 36 workers, gate 0.6, multi-budget 0.5, 240 tasks, 60 reps;
attempt 1 tokens x0.575, turns x0.68, its fixes kept with p = rho): fresh 72.8 ± 3.0 (9% of tasks finish attempt 1);
verifycode rho 0.6 73.4 (19%); rho 0.4 72.8; rho 1.0 74.6; check at 30% of fresh tokens (turns 40%) 76.2 (49%);
with thrash (x0.6 above 0.88) 66.1 vs 66.2. Kaggle runs out of tokens before most attempt 1s, so an informed check
helps only if it is much cheaper than a fresh attempt; next test: `verifycode` with a hard ~20k budget for informed
attempts. Needs a solver change (attempt 1 after attempt 0 of the same task); not in the pinned candidate.

**hybrid / split (run directly).** `split` was the best offline mix (84.5–85.3% at ~25k, picked from the runs above);
run directly it gives 81.5% mean pass@2 over 3 samples vs `ctrl` 83.9% (3 samples), at 13% fewer tokens: the
offline pick was fitted on the same noise. `hybrid`'s audit of train-passing answers fixes less than `verify`.
Neither is better than a fresh attempt.

**Hard cap for informed attempts (sim only, not run).** Capping `verifycode` at 24k (32k) output tokens, from its
recorded attempts (fix kept only if it used <= the cap): tokens x0.39 (0.44), turns x0.62, fixes x0.35 (0.46) of
fresh. `kaggle_sim.py`: 73.6 ± 3.1 (73.6 ± 3.2) vs uncapped `verifycode` 73.6 and fresh 72.8; 34% (28%) of tasks
finish attempt 1. Cheaper checks lose fixes as fast as they free tokens, so attempt-1 changes are worth <= ~1 point
on Kaggle. **Conclusion: keep fresh attempt 1s (the pinned candidate).**

**DeepSeek API replicate** (2026-10-26 10:00 UTC, off-peak, 2 samples each, 0 errors after retrying connection
errors, $9.23): `ctrl` pass@2 84.0 / 85.0% (pass@1 73.2 / 77.9%) at 29.9k tokens; `split` 82.8 / 84.4% (81.1 / 80.3%)
at 25.1k (−16%), fixed / broke 29 / 9 vs ctrl 30 / 33. Both providers together (5 samples each): `ctrl` 84.1%, `split`
82.4% (−1.7 pp, a tie within the 2–3 pp band) at −15% tokens. `split` changes fewer answers both ways, and its later
attempt has a higher pass@1 (79.6% vs 75.2%) because it mostly resubmits attempt 0's answer. Kaggle value <= ~1 point (sim
73.6 vs 72.8), so the pinned candidate keeps fresh attempts.
