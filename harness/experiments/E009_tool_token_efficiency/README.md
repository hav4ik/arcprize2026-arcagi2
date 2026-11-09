# E009: Token efficiency of the Flash tool-use harness

## Goal
Beat **E004-GL** (grid mode, `lean2` prompt, low effort, 120k total / 64k per turn / 24k final, harness `persistent-v3`) on both axes at once, on the full public eval (120 tasks / 167 pairs, ARC Prize task-weighted):
- E004-GL: pass@1 **66.0%** (4 samples: 66.7 / 70.0 / 64.4 / 62.8), pass@2 76.4%, **49.5k** output tokens per attempt (all turns).
- Target: **≤ 39.6k tokens per attempt (−20%)** and **pass@1 > 66.0%**, with ≥ 2 full samples and a paired comparison against E004-GL's samples.
- Why tokens matter: the end goal is SFT data for a small model on Kaggle (4×L4, 12 h, 240 tasks → about 20–25k tokens per attempt affordable), so short successful rollouts are valuable in themselves.

## Where E004-GL's tokens go (no API cost; 668 attempts = 4 samples)

| Outcome | Attempts | Mean output tokens | Share of all output |
|---|---|---|---|
| correct | 431 | 34.8k (median 28.7k) | 45% |
| wrong answer | 121 | 57.6k | 21% |
| no answer | 116 | 95.5k | 34% |

- **No-answer attempts are dead ends.** 28 hit the 120k cap mid-turn and 88 ended in the forced "Finalize now" turn without a parseable grid. 132 forced final turns cost 12.2k tokens each (1.6M in total, 5% of output) and almost never produce a correct grid (E004 trace review: 5 of 231).
- **Re-checking and retyping.** 505 of 552 answers were first printed by a tool cell; the model then spends a mean 3.3k (correct) / 7.8k (wrong) more tokens before answering: re-verification and copying the grid by hand into the reply. That is 2.1M tokens (6%), and copying loses correct answers (dropped or duplicated rows, miscopied cells: +2.5 pp at low effort in the E004 review).
- **Rule discovery dominates.** Reasoning is 90M characters against 7.6M of code. In 10 traces read, most tokens sit in a few 10–40k-token turns where the model reasons by hand over printed grids. Failing attempts cycle through hypotheses for 15–35 turns.
- **Simple caps cost accuracy** (simulated on the 4 samples: an attempt over the cap counts as unanswered):

  | Cap | 50k | 60k | 70k | 80k | 90k | 96k | 120k |
  |---|---|---|---|---|---|---|---|
  | pass@1 | 50.5% | 55.3% | 58.8% | 62.5% | 64.3% | 64.8% | 66.0% |
  | tokens / attempt | 34.7k | 38.6k | 41.8k | 44.4k | 46.4k | 47.4k | 49.5k |

  So the cut has to come from behaviour (answer as soon as the code works, give up cheaply), not from a bare cap.

## Harness changes (harness `persistent-v4`; all opt-in, defaults reproduce `persistent-v3` byte for byte)
Options of `evals/run_deepseek_tools.py`:
- `--answer-tool`: grid mode gets a second tool, `submit_answer(code)`. `code` is an expression evaluated in the model's own session (e.g. `solve(test_input)` or `answer`), or statements that set `answer`. Printing during evaluation is captured. A valid rectangular grid ends the attempt and is the answer; an invalid one returns an error and the attempt continues. The grid never passes through retyped text. The session still never holds test outputs.
- `--final-mode session`: no forced "Finalize now" turn and no 24k reserve. When the budget or the turn limit runs out, or the model stops without a valid grid, the harness reads the session variable `answer`; if it is a valid grid it becomes the answer.
- `--budget-notes F` (with `--budget-warn W`, default 0.8): once F of the total budget is used, every tool output ends with `[budget: 52k of 96k output tokens used]`; from W on it adds "Budget nearly spent: submit your best answer now…". The note sits in the tool message, so no new user turn starts (a user turn makes DeepSeek drop earlier reasoning).
- `--lenient-parse`: if the ARC Prize parser finds no valid grid in the final text, take the last valid JSON grid (trailing commas tolerated), else the last block of one-JSON-row-per-line.
- `--max-balance-stop` (default $20): checks the DeepSeek balance at launch and every 5 min; below it, no new attempts start.
- New prompt suffixes (require `--answer-tool`): `lean2sub` = `lean2` + "when train passes, call submit_answer(solve(test_input)); never retype the grid; keep your best guess in `answer`"; `lean2brief` = `lean2sub` + "think in short notes, test ideas with small experiments, print compact views, submit at once without re-verifying".

**Exactly one scored grid per attempt**, chosen deterministically by how the attempt ended: an accepted `submit_answer` ends the attempt → its grid; else a final text reply → the ARC Prize parse of that text, then (with `--lenient-parse`) the lenient parse; else (budget/turns exhausted, or a text reply without a grid) the session's `answer`. The record's `answer_source` is `submit`, `parse`, `lenient`, `session` or null. Results are reported with and without lenient-parse answers.

Tests: `ARC_SANDBOX_URL=<url> .venv/bin/python -m unittest discover -s evals/tests` (38 tests, including `submit_answer` against the real sandbox).

## Dev set
56 pairs / 49 tasks (`dev_pairs.txt`), a stratified random third (seed 2026) of the 167 pairs by E004-GL's success count over its 4 samples (0/4: 34 pairs in full → 12 in dev; 1/4: 6 → 2; 2/4: 22 → 7; 3/4: 39 → 13; 4/4: 66 → 22). On these pairs E004-GL has pair accuracy 63.8% (full 64.5%) and 47.8k tokens per attempt (full 49.5k), so the dev set is representative. Each variant runs once; the comparison is paired against E004-GL's 4-sample mean on the same pairs (`evals/paired_vs_baseline.py`). One dev sample has about ±5 pp of noise on accuracy; tokens are much more stable.

## Runs

All runs: DeepSeek V4.1-Flash, grid mode, low effort, 64k per-turn cap, off-peak (Sunday 2026-10-11), harness `persistent-v4`, `--answer-tool --final-mode session --lenient-parse`. Launcher: `experiments/E009_tool_token_efficiency/run_variant.sh <tag> <workers> <pairs-file|full> <attempts> [harness args]`. Results: `evals/results/tools_persistent_grid_<prompt>_deepseek-flash_low_<tag>.jsonl`, logs `evals/results/log_<tag>.txt`.

Variant names: `sub` = prompt `lean2sub`, `brief` = prompt `lean2brief`; the number is the hard total budget in k tokens; `n` = budget notes from 50% used (`--budget-notes 0.5`); `a` = budget notes on every tool output (`--budget-notes 0.01 --budget-warn 0.75`); `sNN` = also a soft target of NN k tokens (`--budget-soft`).

### Dev screening (56 pairs, 1 sample each; baseline = E004-GL's 4-sample mean on the same pairs)
Cost estimate before launch: 56 attempts × ~40k output tokens × $0.60/M ≈ $1.35 output + ~15% input ≈ $1.5 per variant. Recorded (`est_cost_usd`, list prices incl. input): $1.38–1.81 per variant.

| run | pass@1 (task-weighted) | E004-GL same pairs | diff [95% CI] | tok/attempt mean / median | vs E004-GL (47.8k) | correct-rollout tok | answer sources | cost |
|---|---|---|---|---|---|---|---|---|
| sub96 | 73.1% | 65.9% | +7.2 [−2.5, +16.4] | 48.5k / 41.2k | +1.6% | 38.8k | submit 47, none 7, session 2 | $1.81 |
| sub96n | 74.1% | 65.9% | +8.2 [−0.3, +16.4] | 45.8k / 38.2k | −4.1% | 37.9k | submit 52, none 4 | $1.70 |
| brief96n | 66.0% | 65.9% | +0.1 [−7.8, +6.9] | 46.2k / 36.1k | −3.3% | 34.6k | submit 52, none 4 | $1.71 |
| brief80n | 65.0% | 65.9% | −0.9 [−13.4, +10.6] | 44.1k / 40.5k | −7.7% | 39.1k | submit 52, none 4 | $1.68 |
| brief80a | 72.8% | 65.9% | +6.9 [−0.8, +14.5] | 41.6k / 38.4k | −12.9% | 34.8k | submit 52, none 4 | $1.52 |
| brief72a | 65.0% | 65.9% | −0.9 [−9.9, +7.1] | 40.4k / 34.3k | −15.5% | 29.8k | submit 48, none 8 | $1.48 |
| brief64a | 58.8% | 65.9% | −7.1 [−16.6, +1.6] | 40.8k / 45.4k | −14.6% | 32.8k | submit 47, none 8, session 1 | $1.49 |
| sub80a | 68.7% | 65.9% | +2.8 [−5.6, +11.2] | 40.1k / 32.2k | −16.1% | 33.0k | submit 54, session 1, none 1 | $1.49 |
| sub72a | 66.3% | 65.9% | +0.4 [−9.0, +9.2] | 40.3k / 37.4k | −15.6% | 33.1k | submit 48, none 7, session 1 | $1.48 |
| **sub64a** | 73.1% | 65.9% | +7.2 [−2.6, +17.1] | **37.2k** / 36.2k | **−22.1%** | 33.2k | submit 51, none 4, session 1 | $1.38 |
| sub80s40 | 71.1% | 65.9% | +5.2 [−4.3, +14.2] | 38.8k / 31.0k | −18.8% | 32.1k | submit 55, session 1 | $1.44 |
| sub72s32 | 75.2% | 65.9% | +9.3 [+1.8, +16.8] | 39.1k / 37.8k | −18.1% | 34.1k | submit 54, none 1, session 1 | $1.44 |
| sub64s32 | 65.6% | 65.9% | −0.3 [−13.0, +11.7] | 38.0k / 35.2k | −20.5% | 30.4k | submit 49, none 7 | $1.39 |

No answer came from the lenient parser in any dev run (every text answer was either absent or parsed by the ARC Prize parser), so accuracy with and without lenient-parse answers is identical.

**Dev findings** (single samples: treat ±6 pp as noise):
1. **The answer hand-off is the accuracy win.** Every `lean2sub` run is at or above E004-GL; the seven `sub` runs average 71.7% against 65.9%. 90–98% of answers come through `submit_answer`, so copy errors are gone, and the model no longer retypes 30×30 grids.
2. **Dropping the forced final turn alone does not save tokens** (sub96 +2%): without the 24k reserve, failing attempts simply spend the whole budget.
3. **Budget notes on every tool output are the main token lever** (−13 to −22% at 64–80k caps vs −4 to −8% with notes from 50%). The model sees `[budget: 23k of 64k output tokens used]` after every cell and converges sooner. Failing attempts shrink most, but correct rollouts also get shorter (≈33k vs 38k).
4. **The brevity prompt (`lean2brief`) hurts accuracy** (brief vs sub at the same budget and notes: −8, +4, −1, −14 pp at 96/80/72/64k) for little extra saving, so it was dropped.
5. A soft target in the notes (`sNN`) saves another ~3% at equal hard cap, with no visible accuracy cost.

### Full-set validation

**sub64a, 2 samples (stopped early).** Estimate before launch: 334 attempts × ~37k × $0.60/M ≈ $7.4 output, ~$8.2 with input. Stopped after 124 records ($3.3 recorded) because it was clearly losing:
- on the 61 pairs with both samples done: pass@1 59.1% vs E004-GL 72.0% on the same pairs, **−12.8 pp, 95% CI [−22.9, −2.7]**, at −22% tokens (40.3k vs ~51.7k);
- on the 17 of those pairs that are in the dev set it was 67.6% vs 76.5% for the dev run of the same config, and on the 43 non-dev pairs 54.7% vs 68.6% for E004-GL;
- the losses are pairs that E004-GL solves with 50–100k tokens (16b78196, 20270e3b, 28a6681f, 2d0172a1, 332f06d7, …): at 64k they either run out of budget or submit early under the "budget nearly spent" note and are wrong.

So the dev set under-represented the long tail of slow-but-solvable pairs, and picking the best of 13 noisy dev runs overstated sub64a (winner's curse). The next full runs use 80k and 96k hard caps with a 40k soft target in the notes, which keeps the long tail reachable while still pushing typical attempts to finish early.

**Full-set results (2026-10-11).** Paired vs E004-GL (4 samples) on all 167 pairs, bootstrap over tasks. Records with API connection errors are excluded; there were 0–3 per run after retries.

| Config | Samples | pass@1 (per sample) | diff vs E004-GL [95% CI] | pass@2 (mean over attempt pairs) | Tokens / attempt, mean / median | vs 49.5k | Correct rollouts: p50 / p90 | Long-tail pairs* | Cost recorded |
|---|---|---|---|---|---|---|---|---|---|
| E004-GL (baseline) | 4 | 66.0% (66.7 / 70.0 / 64.4 / 62.8) | — | 76.4% | 49.5k / 41.4k | — | 28.7k / 72.1k | 71.4% | $22.60 |
| **sub80s40** | 3 | **67.8%** (67.8 / 70.7 / 64.9) | +1.9 [−2.5, +6.1] | **77.4%** (a0+a1 76.9%); +0.9 [−3.7, +5.5] | **40.6k** / 37.5k | **−17.9%** | 25.2k / 60.4k | 63.1% | $14.03 (3 samples incl. retries) |
| sub96s40 | 1 | 67.8% | +1.8 [−5.0, +8.3] | — | 42.1k / 38.2k | −14.9% | 27.3k / 68.7k | 67.9% | $4.69 |
| sub96n | 1 here; 2 in E011: 68.3% / pass@2 78.5% | 69.4% | +3.6 [−2.4, +9.4] | — | 42.8k / 37.1k | −13.1% | 27.3k / 66.8k | 72.7% | $4.68 |
| sub64a | 1 | 58.2% | **−7.8 [−14.3, −1.5]** | — | 38.7k / 40.7k | −21.8% | 24.7k / 49.3k | 44.6% | $5.85 |

\* Long-tail pairs are the 56 pairs whose E004-GL correct rollouts needed more than 40k tokens on average; the column is pair accuracy.

**Was the goal met?** Only partly.
- **sub80s40** (3 full samples):
  - beats E004-GL on accuracy, pass@1 67.8% vs 66.0% and pass@2 77.4% vs 76.4%, but neither difference is significant;
  - cuts tokens per attempt by 17.9% (40.6k), just short of the −20% target (39.6k).
- **sub64a** is the only config that clears −20%, and it loses 7.8 pp. It truncates the long tail: long-tail pairs drop from 71% to 45%.
- On these runs the accuracy-vs-token trade-off is steep below about 40k (see E010).

**What sub80s40 is:** grid mode, low effort, `lean2sub` prompt, the `submit_answer` tool, no forced final turn (session `answer` fallback), and a budget note on every tool output. The hard cap is 80k and the note states a 40k soft target.
```bash
cd evals && ../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool \
  --final-mode session --lenient-parse --max-tokens 64000 --max-total-output 80000 \
  --budget-notes 0.01 --budget-warn 0.75 --budget-soft 40000 --price-multiplier 1 --attempts 1
```

**What changed for SFT data:**
- Correct rollouts are shorter. Against E004-GL on the same pairs, the p50 falls from 28.7k to 25.2k and the p90 from 72.1k to 60.4k; 50% of correct sub80s40 rollouts are ≤25k.
- Every answer is handed over by `submit_answer(expr)` instead of being retyped, so there are no copy errors.
- No answer came from the lenient parser in any run, so accuracy with and without it is identical.

**Dev vs full.** The 56-pair dev set overstated the lean2sub family:
- on dev, the seven `sub` runs averaged +5.8 pp; on the full set they scored −7.8 to +3.6;
- sub64a went from +7.2 on dev to −7.8 on the full set.

Picking the best of many noisy single-sample dev runs inflates the winner. Only full-set samples should be trusted.

**Safety** (5 sub80s40 traces read: a47bf94d.0, 269e22fb.1, dd6b8c4b.1, 195c6913.0, 6ffbe589.0; plus a pattern scan of all 2,124 sample-1 cells and the dev traces):
- Nothing touches the host, the network or other sessions.
- Odd but harmless:
  - a dead `exec(open('/dev/stdin').read()) if False else None` line (a47bf94d, 3a25b0d8);
  - one cell that wiped `train` by accident (`train=[... for x in []]`, d59b0160); the model then retyped the grids by hand and still solved the pair.
- One quality issue: in a47bf94d the model hand-patched cells of its `answer` array before submitting, against the prompt's "never construct grids by hand". Filter such traces for SFT.

**Spend:** E009 recorded $54.97. That is $20.0 for dev screening and $35.0 for full-set runs, including the sub96n second sample ($5.5) run during E011. Cost estimates before each launch are given in the sections above, and recorded costs matched them within ~10%. The balance reconciliation for E009–E011 is in `experiments/E011_pareto_tool_harness/README.md`. Follow-ups: E010 (`experiments/E010_half_reasoning`) and E011 (`experiments/E011_pareto_tool_harness`), whose recommendation supersedes the one above.
