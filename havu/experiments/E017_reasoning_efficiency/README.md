# E017: reasoning efficiency of the Flash tool harness (failure analysis, then targeted harness changes)

## Plan
- **Phase 1** ($0, offline): where DeepSeek V4.1-Flash spends its tokens in the best tool harness (sub80s40 / sub96n), why it fails, and which compute-allocation policies a train-check verifier supports.
  - 1A: `policy_sim.md`, train-check labelling and policy simulation.
  - 1B: `failure_analysis.md`, replay of all 835 attempts plus 48 annotated traces.
- **Phase 2** (paid, $40 approved): implement the top three proposals of phase 1B and the policy that phase 1A recommends, then measure each against sub80s40's 3 full-set samples.

## Phase 1 summary
- **Tokens go to prose discovery.** About 70% go to reading grids and searching for hypotheses in prose before code tests them. Readers judged about 42% of all tokens wasted.
  - Long single turns predict failure: an attempt whose longest turn is 30k or more is 26% correct, against 81% under 10k.
  - Hand-transcribed grid rows track failure too.
- **Failures rarely miss the core idea.** The idea appears at a median 16% of the attempt. What goes wrong is a sub-rule, a role binding, or a test case the training pairs don't show.
  - The largest failure mode is train_overfit (12/28): the rule fits every training pair and is wrong on the test.
  - 24/28 failing attempts had perception errors.
  - Contradictions were set aside in 23/28 failing attempts, against 2/20 correct ones.
- **The train check is a useful selector but a poor gate.** P(correct | train_pass) is 86% and P(correct | train_fail) is 36%.
  - Gating the second attempt on it loses pass@2.
  - Policy (f), 2 attempts plus a 3rd only when neither passes train, simulated at **+2.0 pp pass@2 [+1.0, +3.2]** for +23% tokens.
  - Restarting attempts that have no train pass by X tokens loses 7–12 pp.
- **Proposals ranked for phase 2:**
  - P1: computed perception pack;
  - P2: roles-first prompt with a test audit;
  - P3: session helpers with `check(solve)`;
  - plus the conditional 3rd attempt.

## Phase 2: what was built (harness `persistent-v5`, all opt-in; defaults unchanged)
| Flag | What it does | Code |
|---|---|---|
| `--task-summary` (P1) | Host-computed perception pack appended to the first user message. Per pair: a header (sizes, background, changed cells and colour transitions, colour counts, whole-grid relations such as D8, crop, scale and tile). An input object table: 8-connected same-colour components with size, bbox, holes, border and task-wide shape letters invariant under D8, at most 12 rows with the rest summarised per colour. Multi-colour components and separator panels. In→out matching: unchanged, moved (dy,dx), recoloured, transformed (D8 op), removed, changed, and unmatched output objects. For the test input: the same table plus an "Outside train range" line covering size, new or absent colours, background, colour count, object count, border contact, object sizes and unseen shapes. Deterministic. On the 167 eval pairs it is 1.7–9.7k characters, mean 4.7k (about 1.5k tokens). | `evals/task_summary.py` |
| `--prompt lean2roles` (P2) | `lean2sub` plus the P2 text of `failure_analysis.md`, verbatim: a ROLES note before solve(), no "noise/irrelevant" labels without code evidence, and a printed test-vs-train difference audit after train passes; never submit while a training pair fails. | `evals/run_deepseek_tools.py` |
| `--helpers` (P3) | `evals/arc_helpers.py` is executed in the session prelude, so it is also restored after a state reset. It provides `objects`, `bbox`, `crop`, `bg_colour`, `d8`, `match_d8`, `panels`, `stamp` (with clipping), `ray`, `show`, and `check(fn)`, which prints at most 8 differing cells per pair and then ALL PASS. About 300 tokens of documentation go in the run_python description, plus the prompt line "Use check(solve) after every change to solve." | `evals/arc_helpers.py` |
| `--train-check` | After the attempt ends, labels the answer program in the same session (paths expr → cell → fn → transcript replay, as `label_train.py` does) and records `train_pass`, `train_frac`, `train_trace` and `train_paths`. No model tokens; only hashes leave the sandbox. **Validated:** it agrees with the phase-1 labels on 40/40 replayed sub80s40 attempts. | `evals/train_check.py` |
| `--extra-attempt-if-no-train-pass` | With `--attempts 2`: when attempts 0 and 1 of a pair have both finished and neither passes train, attempt 2 is scheduled (3 at most). It is not in the config, so it can be added on resume. Final answers come from `ranked_answers`: by train_pass, then attempt order, top-2 distinct. | `evals/run_deepseek_tools.py`, `evals/train_check.py` |
| `--auto-peak-price` | Prices each turn at 2× when its request starts inside a weekday peak window (01–04 and 06–10 UTC). Records `est_cost_peak_usd` and `peak_turns`. | `evals/run_deepseek_tools.py` |

Tests: `ARC_SANDBOX_URL=http://172.18.0.2:6000 .venv/bin/python -m unittest discover -s evals/tests` (87 tests in all; `evals/tests/test_e017.py` has 20, 5 of them against the real sandbox).

## Phase 2 runs
All runs use the full public eval (120 tasks / 167 pairs) with the sub80s40 budgets: grid mode, low effort, `submit_answer`, session fallback, lenient parse, 80k hard / 40k soft, budget notes on every output, 64k per turn, `--train-check --auto-peak-price`.

| Run | Config | Samples | When | Estimate | Recorded cost |
|---|---|---|---|---|---|
| smoke | V3 config, 3 tasks (5 attempts) | 1 | 06:15 UTC (peak) | $0.1 | $0.06 |
| V1 `e017_full_v1sum` | `lean2sub` + `--task-summary` (P1) | 1 | 06:19–08:14 (peak, 2×) | ~$9.4 at peak | $8.26 |
| V2 `e017_full_v2roles` | `--prompt lean2roles` (P2) | 1 | 06:19–08:19 (peak) | ~$9.4 | $9.49 |
| V3 `e017_full_v3all` | `lean2roles` + `--task-summary` + `--helpers` (P1+P2+P3) | 1 | 06:19–08:14 (peak) | ~$9.4 | $8.98 |
| V3, sample 2 + policy | same file; `--attempts 2 --extra-attempt-if-no-train-pass` | +1, +24 extra attempts | 10:01–10:55 (off-peak) | ~$4.7 + $1–2 | $4.31 + $1.05 |

- Round 1 ran at peak prices on the user's direction ("no need to wait for off-peak").
- Round 2 was held until 10:01 UTC. At peak it would have cost about $11 and broken the $38 cap.
- The optional runner-up sample was dropped: only $5.8 was left under the cap, against the $9 margin the plan required.

Result files are `evals/results/tools_persistent_grid_{lean2sub,lean2roles}_deepseek-flash_low_e017_full_{v1sum,v2roles,v3all}.jsonl`, with logs in `evals/results/log_e017_full_*.txt`.

Commands (from `evals/`):

```bash
COMMON="--mode grid --effort low --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 80000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 40000 --price-multiplier 1 \
  --auto-peak-price --train-check --max-balance-stop 45 --sandbox-url http://172.18.0.2:6000"
../.venv/bin/python run_deepseek_tools.py $COMMON --attempts 1 --workers 6 --prompt lean2sub --task-summary --tag e017_full_v1sum
../.venv/bin/python run_deepseek_tools.py $COMMON --attempts 1 --workers 6 --prompt lean2roles --tag e017_full_v2roles
../.venv/bin/python run_deepseek_tools.py $COMMON --attempts 1 --workers 6 --prompt lean2roles --task-summary --helpers --tag e017_full_v3all
# round 2: the winner's 2nd sample and the conditional 3rd attempt, in one resume of the same file
../.venv/bin/python run_deepseek_tools.py $COMMON --prompt lean2roles --task-summary --helpers --tag e017_full_v3all \
  --attempts 2 --extra-attempt-if-no-train-pass --workers 16
```

## Phase 2 results
Produced by `scripts/phase2_compare.py`; full numbers are in `phase2_results.json` (and `phase2_round1.json`).
- **Kaggle pass@1** is the mean over samples of pair accuracy over the 167 test pairs. The task-weighted (ARC Prize) version is also shown.
- **Δ** is paired against the mean of sub80s40's 3 samples on each pair. The 95% CI comes from a 10k-resample cluster bootstrap over the 120 tasks.
- **sub80s40 reference:**
  - per-sample pair-weighted pass@1 is 66.5 / 69.5 / 63.5% (mean 66.5%), and 67.8% task-weighted;
  - pass@2 is 76.4% pair-weighted (attempt pairs 76.0 / 76.0 / 77.2%) and 77.5% task-weighted;
  - 40.7k output tokens per attempt, $4.50 per sample off-peak.

| Variant | Samples | Kaggle pass@1 (pair-w) | Δ vs sub80s40 [95% CI] | pass@1 task-w (Δ [CI]) | pass@2 pair-w (Δ [CI]) | pass@2 task-w | pass@2 + policy, pair-w / task-w | Out tok/attempt mean / median | $ recorded (off-peak equivalent) |
|---|---|---|---|---|---|---|---|---|---|
| sub80s40 (baseline) | 3 | 66.5% | — | 67.8% | 76.4% | 77.5% | — | 40.7k / 37.6k | $4.50 |
| V1 summary | 1 | 67.1% | +0.6 [−5.9, +7.1] | 66.0% (−1.8 [−8.2, +4.6]) | — | — | — | **37.1k** / 31.0k (−9%) | $8.26 ($4.13) |
| V2 roles | 1 | 67.1% | +0.6 [−5.5, +6.7] | 67.8% (+0.0 [−6.5, +6.4]) | — | — | — | 42.9k / 42.2k (+5%) | $9.49 ($4.75) |
| **V3 all** (round 1) | 1 | **70.1%** | +3.6 [−2.3, +9.4] | 69.0% (+1.3 [−4.9, +7.2]) | — | — | — | 39.2k / 33.8k | $8.98 ($4.49) |
| **V3 all** (both samples) | 2 | **69.2%** (70.1 / 68.3) | **+2.7 [−2.5, +7.7]** | 68.3% (+0.6 [−4.7, +5.6]) | 77.2% (+0.8 [−4.4, +5.8]) | 76.0% (−1.5 [−7.2, +3.9]) | **78.4%** (+2.0 [−3.3, +7.2]) / 77.2% (−0.3 [−5.8, +5.1]) | 39.5k / 35.0k (−3%) | $13.29 + $1.05 policy |

Pairwise among the round-1 samples (pair-weighted pass@1):
- V3 − V1: +3.0 [−3.5, +9.6] (18 pairs won, 13 lost);
- V3 − V2: +3.0 [−4.5, +10.6];
- V2 − V1: +0.0.

None of these differences is significant. One sample swings by ±3 pp: sub80s40's own samples span 63.5–69.5%.

### The conditional 3rd attempt (policy f), live
- **When it fired:** on 24 of 167 pairs (14%), where neither attempt 0 nor attempt 1 passed the train check.
- **Cost:** +12% output tokens per pair (79.0k → 88.8k; the extra attempts average 67.9k), and $1.05.
- **What the extra attempts produced:**
  - 5 were correct and 3 passed train; 3 of those were correct.
  - They rescued 2 pairs. The other correct extras landed on pairs that attempt 0 or 1 had already solved without passing train.
- **Net:** pass@2 rose from 77.2% to **78.4% (+1.2 pp [+0.0, +3.0])** pair-weighted, and from 76.0% to 77.2% task-weighted.
- **Against phase 1A:** the simulation predicted +2.0 [+1.0, +3.2] for +23% tokens. The live gain is a little smaller at half the extra tokens, and consistent with it.
- **Train-check quality,** on the 334 V3 attempts:

  | Quantity | V3 (all three) | V1 | V2 | Phase 1 |
  |---|---|---|---|---|
  | Attempts that pass train | 259 | — | — | — |
  | P(correct \| train_pass) | 84% | 81% | 86% | 86% |
  | P(correct \| train_fail) | 27% | 44% | 26% | 36% |

  - Labels came from cell 249, expr 42, transcript 12 and fn 7; 24 attempts had no answer, and 0 were untraceable.

### Mechanism checks (per attempt, from the traces; same regexes for the baseline)

| | sub80s40 | V1 summary | V2 roles | V3 all |
|---|---|---|---|---|
| Hand-transcribed grid lines in reasoning (mean) | 30.6 | **19.2** | 28.0 | 21.1 |
| Attempts with ≥10 / ≥40 such lines | 39% / 19% | **28% / 16%** | 41% / 16% | 33% / 15% |
| Longest single turn, median / p90 | 11.8k / 28.7k | 12.4k / 27.5k | 13.1k / 28.0k | 13.4k / 29.2k |
| Attempts with a turn ≥30k | 8% | 8% | 8% | 9% |
| ROLES note present | 0% | 1% | 46% | 50% |
| Test-vs-train difference audit (regex) | 30% | 26% | 49% | 37% |
| "noise / irrelevant / distractor" in reasoning | 53% | 43% | 56% | 50% |
| Summary referenced in reasoning | 2% | 91% | 7% | 87% |
| Helpers used / `check(` calls per attempt | 5%\* / 0 | 12%\* / 0 | 10%\* / 0 | 30% / 1.1 |
| No answer | 6.0% | 5.4% | 7.8% | 7.2% |
| Correct-rollout tokens, p50 / p90 | 25.4k / 62.8k | **22.5k** / 58.6k | 28.7k / 64.9k | 24.6k / 61.7k |

\* In runs without `--helpers`, the regex counts self-written functions with helper-like names (`crop(`, `objects(`).

**What the mechanism checks show:**
- **P1 (summary) does what it was meant to.**
  - The model reads it in 91% of attempts.
  - Hand transcription falls by about a third.
  - Tokens fall 9%, and correct rollouts get shorter (p50 22.5k).
  - Accuracy did not move: +0.6 pp pair-weighted, −1.8 task-weighted.
- **P2 (roles prompt) is followed about half the time.** ROLES notes appear in 46–50% of attempts and the test audit in 49%.
  - It does not stop "noise" labelling (56%).
  - It costs +5% tokens and 0.7 more turns per attempt (13.6 against 12.9).
  - Alone it gains nothing.
- **P3 (helpers):** used in 30% of V3 attempts, with `check()` called about once per attempt.
- **None of the three shortens the long prose turns.** Longest-turn p90 stays at about 28–29k and about 8% of attempts still have a turn of 30k or more, which phase 1 identified as the strongest failure signature. A per-turn cap (proposal P5) is the untested lever left for that.

## Recommendation
1. **Use V3 (`--prompt lean2roles --task-summary --helpers`) with `--train-check` as the new default candidate. It is not a proven improvement.**
   - It has the best pass@1 of the four configs: 69.2% over 2 samples, +2.7 pp pair-weighted [−2.5, +7.7].
   - It uses 3% fewer tokens than sub80s40.
   - Pass@2 is level: 77.2% against 76.4% pair-weighted, but 76.0% against 77.5% task-weighted.
   - Nothing is significant at 1–2 samples. The honest summary is "no harm, slightly fewer tokens, a possible +2–3 pp on pass@1".
2. **Use the conditional 3rd attempt whenever budget allows.** It is the only lever with a CI that excludes a loss: +1.2 pp pass@2 [+0.0, +3.0] for +12% tokens. The train check that drives it costs no model tokens. It also gives Kaggle a principled answer order: train_pass first.
3. **For a Kaggle token budget, P1 alone is the efficiency lever:** −9% tokens and shorter correct rollouts at equal accuracy.
   - It is host-side and deterministic, so it ports directly to the Qwen/vLLM harness.
   - The summary adds about 1.5k prompt tokens (mean 4.7k characters, cap 10k characters).
4. **Do not adopt P2 (lean2roles) on its own.** It costs +5% tokens for no measurable gain; inside V3 its cost is offset by P1.
5. **Next steps, if pursued:**
   - a second sample of V1 and of V3 at 1× price (about $4.5 each) to tighten the CIs;
   - P5 (cap the per-turn tokens at about 24k), because none of P1–P3 touched the long-turn signature.

## Spend and balance
| Item | Estimate | Recorded |
|---|---|---|
| Smoke test | $0.1 | $0.06 (peak) |
| Round 1 (3 variants × 167, peak 2×) | about $28 | $26.73 |
| Round 2 (V3 sample 2, off-peak) | $4.7 | $4.31 |
| Policy extras (24 attempts, off-peak) | $1–2 | $1.05 |
| **E017 total** | — | **$32.16**, of which $26.79 at peak; $40 approved, $38 launch cap |

The balance went from $82.71 to $50.61, a drop of $32.10, which matches the recorded total. No launch happened below the $45 floor.
