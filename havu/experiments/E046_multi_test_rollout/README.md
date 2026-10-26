# E046: one rollout per task for tasks with several test inputs (DeepSeek API, Flash-1017)

**Why.** The private test set has ~335 test pairs over its 240 tasks (a teammate's LB probe, 2026-10-25), the eval
set's ratio (120 tasks, 172 pairs). The Kaggle solver runs separate attempts per (task, pair), so multi-pair tasks
cost one full attempt per test input: on E128's eval run they are 45 of 120 tasks but 92 of 167 pairs and 59% of all
tokens (E042 README, "The private test set has ~335 pairs"). Submit v3 scored 50.9, which fits running out of time
under KV thrash. The user's idea, as in the arc2-2026-harness-8d63d5 notebook: show every test input in one
rollout and answer all of them with one program.

**Harness option `--multi-test`** (evals/run_deepseek_tools.py, opt-in; `multi_test` in the config, harness
persistent-v6; DeepSeek API path only):
- tasks with one test input are unchanged; a task with n > 1 gets one attempt per attempt index (job (task, 0, a));
- the prompt is the ARC Prize prompt with every test input numbered (`--Test input i--`) plus a note: n test
  inputs, preloaded as `test_inputs` (test_input = test_inputs[0]); one rule must work for all; submit a list,
  e.g. `[solve(t) for t in test_inputs]`, and keep `answer` as that list;
- `submit_answer(code, n)` accepts only a list of n valid grids; the session fallback reads `answer` the same way,
  and `--fallback-solve` evaluates `[solve(t) for t in test_inputs]`; text answers are not parsed;
- one record per pair (pred = that pair's grid, correct per pair); the attempt's usage and cost are on pair 0's
  record, the other pairs carry `multi_share: true` and zero usage; `multi_test: n` on every record;
- `--train-check` labels the program behind test input 0's grid.
Tests: evals/tests test_deepseek_tools + test_e017 pass (defaults unchanged).

**Run** (2026-10-25 21:49 UTC, off-peak): the 45 eval tasks with 2–3 test inputs (92 pairs), 2 attempts, E030's
sub80s40 + header flags exactly (baseline `e030_hdrmin`, same 92 pairs, per-pair attempts):

```
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  --model deepseek-flash --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --attempts 2 --tasks <45 multi-pair ids> --multi-test --auto-peak-price \
  --tag e046_multi_hdrmin --sandbox-url http://172.18.0.6:6000 --workers 48
```
Compare: `evals/multi_test_compare.py <e046 jsonl> <e030_hdrmin jsonl>`.

## Results (2026-10-25 22:02 UTC, 90 attempts, 0 errors, $2.34 recorded)

Same 45 tasks / 92 pairs, 2 attempts each (`evals/multi_test_compare.py`):

| | pass@1 (pairs) | pass@2 (pairs) | output tokens per task per attempt | turns per task per attempt | cost |
|---|---|---|---|---|---|
| per-pair attempts (E030 hdrmin) | 72.8% | 85.9% | 60.0k (median 51.6k) | 30.5 | $3.81 |
| **one rollout per task (`--multi-test`)** | **73.9%** | **82.6%** | **35.5k (median 34.9k), −41%** | **18.3, −40%** | **$2.34** |

Paired pass@2: both 71, per-pair only 8, multi only 5, neither 8 (sign test p ≈ 0.6: a tie within the noise band).
87 of 90 attempts submitted a list; 3 ended without an answer. One rollout per task keeps pass@1 and cuts the
multi-pair tasks' tokens and turns by ~40%. Multi-pair tasks are ~59% of E128's eval tokens, so on a ~335-pair
private set this removes ~24% of all tokens and ~28% of attempts (480 instead of ~670).

**Kaggle estimate** (E042 `deadline_sim.py --multi-scale 0.59`, E128 + v4 server settings, 335 pairs, KV ratio 0.83):
per-pair × 36 71.8 → one rollout per task × 36 75.6, × 44 77.9 (77.1 with KV thrash), × 48 78.7 (76.2). Assumes E128
keeps its accuracy under `--multi-test` as Flash-1017 did. Next: the Kaggle solver (per-task jobs, per-pair records),
`--multi-test` on the self-hosted `--token-exact` path, and an E128 check on a 4×L4 pod.

## Kaggle candidates (2026-10-25 22:48 UTC)

- Solver `kaggle/dsv41_tools/solver_dsv41.py`: opt-in `ARC_MULTI_TEST=1` (pair 0's job per task answers every
  test input; `split_multi` turns the attempt into one record per pair, train label shared) and `ARC_TASK_IDS`
  (smoke-test task list). Mock-tested (args, record split, submission); defaults unchanged.
- Kit dataset **chankhavu/dsv41-tools-kit-mt** (new, private; differs from the v3 kit only in run_deepseek_tools.py
  `--multi-test`, train_check.py opt-in options, and the solver's opt-in multi-test + KV gate).
- Notebooks (v4 settings + `ARC_MULTI_TEST=1`; commit run = smoke on 25ff71a9 (2 tests), 27a28665 (3), 00576224):
  **chankhavu/dsv41-tools-submit-mt** (44 workers) and **chankhavu/dsv41-tools-submit-mt36** (36 workers), Version 1.
- Not checked: E128's own accuracy with one rollout per task (only Flash-1017 on the API).

## E128 on 4x RTX PRO 6000 (2026-10-25/26, self-hosted)

**Question.** Does tool E128 2-bit (the Kaggle model, `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`) keep
its accuracy with one rollout per task, as Flash-1017 did on the API?

**Answer: yes, with a larger budget for multi-test rollouts.** At one 80k budget per task, E128 loses ~9 pp pass@1 on
the multi-test tasks (runs 1-2: a quarter of the attempts run out of budget or turns before answering). With
`--multi-budget 0.5` it matches per-pair attempts on pass@1 and beats them on pass@2 at ~70% of the tokens (runs 3-5,
6 attempts pooled: 67.8% / 81.2% vs 69.0% / 78.3%). The final harness on all 120 tasks (run 7) scores task-weighted
73.7% / 84.6% vs per-pair E128's 72.8% / 82.5%, at −17% tokens per task-attempt overall and −30% on multi-test tasks.

**Setup** (Vast 57732060, 4x RTX PRO 6000 Blackwell Workstation, sm120, 96 GB; 1.5 TB RAM; 22 min from first login
to the eval start). Scripts in `scripts/`:
- `box_install_sm120.sh`: E042 `box_install.sh` steps 1-2 (uv venv from `requirements.box.freeze` + the fork wheels
  `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` / flashinfer vision2, sha256-checked), then the Kaggle
  overlay `kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz` extracted over site-packages and sha256-checked (12 files).
  Nothing else: no L4 pieces (sm89 JIT cache, PR #154, synthetic metadata), the image's CUDA 13.0 toolkit. 14 min.
- `box_dl_e128t.sh`: the HF checkpoint to local disk, 131.09 GB in 14 min (in parallel with the install).
- `serve_e128.sh`: Kaggle's E128 command (`solver_dsv41.py` server_cmd/env) at TP4 with E043 `serve.sh`'s sm120 env
  (`FLASHINFER_CUDA_ARCH_LIST=TORCH_CUDA_ARCH_LIST=12.0`, `TRITON_PTXAS_BLACKWELL_PATH`, V2 runner), Humming GPTQ env,
  fp8 KV, block 128, max-model-len 163840, `--max-num-seqs 128 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90`, prefix caching (+ retention interval 128), Engram `cpu_offload`, FULL graphs. No
  Humming dense linear backend, no `DSV41_INDEXER_Q_PREDECODE`. Startup 6 min (weights 16-48 s, TileLang JIT and
  graph capture ~4 min). 16.92 GiB weights per GPU, 65.6 GiB KV per GPU: **20.67M KV tokens**.
- The image's default vLLM service (Qwen3.5-9B on port 18000, all 4 GPUs) had to be stopped first
  (`supervisorctl stop vllm`).
- `kvmon.sh`: server metrics every 30 s into `/root/dsv41/runs/e128_mt/kvmon.log` on the box.
- Eval: local harness through an SSH tunnel, sandbox `arc-sandbox-e042`, E043 `eval_arc.sh` PRESET=sub80s40
  (`--rescue-code --attempts 2 --multi-test`, 90 workers), 45 multi-pair eval tasks / 92 pairs. Baseline: E128 per-pair
  attempts with the same harness (E043, 8x box), `tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80.jsonl`.

**Run 1** (`e043_e128t_mt_sub80`, 23:14-00:13 UTC, 0 errors; harness as in the kit-mt Kaggle candidate):

| | pass@1 (pairs) | pass@2 (pairs) | output tokens per task per attempt | turns per task per attempt |
|---|---|---|---|---|
| per-pair attempts (E043 baseline) | 69.0% | 78.3% | 77.8k (median 76.4k) | 36.4 |
| **one rollout per task** | **60.3%** | **76.1%** | **47.8k (median 48.7k), −39%** | **21.4, −41%** |

- Paired pass@2: both 62, per-pair only 10, multi only 8, neither 12: pass@2 is a tie, but pass@1 drops 8.7 pp.
  Task-weighted (harness log): pass@1 60.0%, pass@2 76.7%.
- **The loss is no-answer attempts: 24 of 90** (API Flash: 3 of 90). 19 hit the 80k output budget and 5 the 40-turn
  cap; 23 never set `answer`, 1 had `answer` as a list of numpy grids (the bug below). 13 of the 24 task-attempts had
  at least one pair solved by the baseline's same attempt. When a multi attempt submits, 81.6% of its pairs are right.
  The per-pair baseline spends 77.8k tokens per task-attempt in total, so one 80k budget per task is tight for E128,
  which reasons ~1.3x longer than the API model.
- 66/90 submitted a valid list (all via `submit_answer`); 0 rescues; 16.6% of tool calls raised Python errors.
- **Harness bug (fixed after run 1 started):** `submit_answer(code, n)` rejected a list of numpy grids, the prompt's
  own example `[solve(t) for t in test_inputs]` ("only length-1 arrays can be converted", from `json.dumps`). The
  model recovers next turn with `.tolist()`, at the cost of a turn: 15/16 early E128 attempts and 76/90 attempts of the
  API run above hit it. Fix (`evals/run_deepseek_tools.py`, multi only, single-test cells unchanged): nested arrays
  are converted before `json.dumps`; test `DockerTests.test_multi_submit_numpy_grids`.
- Server: ~1.2k gen tok/s average (90 → 40 streams), prefix hits 96.1%, KV peak 4.6%, 0 preemptions.

**Run 2** (`e043_e128t_mt2_sub80_tx`, 00:22-01:44 UTC, 0 errors): the fixed harness plus the Kaggle solver's
`--token-exact --fallback-solve`. pass@1 58.7%, pass@2 69.6%, 46.5k tokens and 19.3 turns per task-attempt
(task-weighted 58.9% / 68.9%).
- Paired pass@2 vs the baseline: both 60, per-pair only 12, multi only 4, neither 16. Against run 1: both 56, run 1
  only 14, run 2 only 8: the two runs differ by sampling noise and fail the same way.
- No-answer again 24/90: 18 at the 80k budget, 6 at 40 turns. None of them had set `answer` or defined `solve`, so
  `--fallback-solve` rescued nothing; one more attempt was saved by the session `answer` at the turn cap.
- `token_exact` over 1737 turns: 567 breaks (turns whose re-rendered text would not have given back the sampled ids,
  i.e. prefix-cache breaks the token-exact path avoided), 0 rerenders, resyncs or parse fallbacks.
- Runs 2 and 3 shared the server (128 running + ~45 queued), so each was ~1.5x slower than run 1.

## Prompt formats (`--multi-prompt`, DeepSeek API, 2026-10-26 01:40-02:35 UTC, peak price)

New opt-in option of `evals/run_deepseek_tools.py`, default `note` (= `MULTI_NOTE`, byte-identical prompt and config):
`lean` (a one-line note), `early` (`note` + "define solve as soon as you have a rule and set `answer` after every
fix"), `first` (only test input 0 in the prompt; all of them in `test_inputs`). Launcher `scripts/run_prompts.sh
<variant> [all]` (sub80s40 + header + the Kaggle candidate's `--multi-test --multi-budget 0.5 --fallback-solve
--train-check --turns-note 5`, sandbox `arc-sandbox`); comparison `scripts/prompt_compare.py`.

45 multi-test tasks / 92 pairs, 2 attempts, 0 errors (per task-attempt: output tokens, input tokens summed over turns):

| run | pass@1 (pairs) | pass@2 (pairs) | pass@2 (tasks) | no answer /90 | out tok mean / median | in tok | turns | $ (2x peak) |
|---|---|---|---|---|---|---|---|---|
| per-pair (E030 hdrmin) | 72.8% | 85.9% | 86.3% | 2 | 60.0k / 51.6k | 905k | 30.5 | 3.81 (off-peak) |
| E046 run above (`note`, no numpy fix / multi-budget) | 73.9% | 82.6% | 83.0% | 3 | 35.5k / 34.9k | 691k | 18.3 | 2.34 (off-peak) |
| **`note`** | **77.7%** | **89.1%** | **89.6%** | **0** | **38.0k / 31.6k** | **771k** | **18.9** | 4.82 |
| `lean` | 76.1% | 84.8% | 85.2% | 0 | 36.3k / 29.7k | 682k | 17.8 | 4.55 |
| `early` | 75.0% | 84.8% | 85.2% | 0 | 37.6k / 29.9k | 697k | 18.1 | 4.72 |
| `first` | 73.4% | 85.9% | 86.3% | 1 | 38.0k / 29.7k | 760k | 19.2 | 4.83 |

- **The wording is a tie.** Two `note` runs differ by 6.6 pp task pass@2, so the noise is about ±5 pp; output tokens
  differ by under 5%. Keep `note`, the harness E128 was validated with.
- `first` saves no input: the model prints the hidden test inputs, and its input per task-attempt is about `note`'s.
- `early` adds nothing once the budget is scaled: 359/360 attempts submitted a list, with no budget-stop no-answers.

**All 120 eval tasks** (`e046p_note_all`: the 45 multi-test tasks above + the 75 single-test tasks, same flags;
240 attempts, all submitted, 0 errors, $10.11 recorded at peak):

| | task pass@2 | pass@1 | no answer | out tok per task-attempt | turns |
|---|---|---|---|---|---|
| per-pair E030 hdrmin, 120 tasks | 84.9% | 74.5% | 6 | 38.2k | 19.4 |
| **`--multi-test` (`note`), 120 tasks** | **88.6%** | **77.6%** | **0** | **29.6k (−23%)** | **15.2** |
| of which single-test tasks (75) | 88.0% (base 84.0%) | 77.3% (75.3%) | 0 (4) | 24.6k (25.1k) | 13.0 |
| of which multi-test tasks (45) | 89.6% (base 86.3%) | 78.1% (73.1%) | 0 (2) | 38.0k (60.0k) | 18.9 |

The single-test path is unchanged in tokens. Its no-answers drop from 4 to 0 with `--fallback-solve --turns-note 5`,
which the E030 baseline did not use. Screen + full run cost $24.10 by balance (peak prices).

## E128 runs on 4x RTX PRO 6000 (2026-10-26, Vast boxes; per-pair baseline = E043 e128t_w10_u2_gptq_eng64_sub80)

Same 45 multi-pair eval tasks / 92 pairs, 2 attempts, E043's eval_arc.sh PRESET=sub80s40 + `--rescue-code`:

| run | harness | pass@1 | pass@2 | no answer | tokens / task-attempt | turns |
|---|---|---|---|---|---|---|
| per-pair baseline (E043) | — | 69.0% | 78.3% | — | 77.8k | 36.4 |
| run 1 | kit-mt as pushed (numpy-list bug), no token-exact | 60.3% | 76.1% | 24/90 | 47.8k | 21.4 |
| run 2 | fixed + `--token-exact --fallback-solve` | 58.7% | 69.6% | 24/90 (18 budget, 6 turns) | 46.5k | 19.3 |
| **run 3** | run 2 + **`--multi-budget 0.5`** (89/90) | **71.3%** | **83.1%** | 8/89 | 54.5k (−29%) | 23.8 |

- Run 2's no-answer attempts never defined `solve`, so `--fallback-solve` rescued none: E128 needs the bigger budget
  in multi mode (one 80k budget for n inputs, where per pair it spent ~78k summed). Run 3's paired pass@2 vs the
  baseline: both 63, baseline only 6, multi only 11, neither 9.
- Numpy-list fix (sub-agent): `submit_answer(code, n)` converts nested numpy grids (the prompt's own example
  `[solve(t) for t in test_inputs]` failed with "only length-1 arrays"); it cost a turn in most attempts.
- `--multi-budget f` (new, opt-in): output budget, soft budget and turn cap × (1 + f (n − 1)) for multi tasks.
- Flash-1017 API screen of prompt wordings (other session, tags e046p_*): all tie; with `--multi-budget 0.5` on all
  120 tasks: task pass@2 88.6% vs per-pair 84.9%, −23% output tokens, 0 no-answer.

## KV pressure on 4xL4 at Kaggle's KV (pods, 1.557M tokens, full Kaggle pipeline on the eval set)

- per pair × 36 (= submit Version 5): KV peaked 0.643 at 2.0 h, 0.54–0.61 to 2.5 h, 290–320 tok/s: stable.
- multi-test × 44 (80k budget): thrashed in bursts from 2.0 h at KV 0.82–0.93 (down to 48 tok/s, hits 0.16).
- multi-test × 52 + KV gate 0.75: the gate held admissions from 1.22 h (50–51 running), but the running contexts kept
  growing: KV 0.895 at 1.65 h, thrash starting. The gate limits admissions, not the KV already held.
- Multi-test streams hold more KV than per-pair ones (all test inputs, longer attempts): fewer streams per pool.
- KV CPU offload is not usable with V4.1 on this fork (E042 README).
- Candidates under test (from ~02:10): multi-test + budget 0.5 at 36 (pod 6), 40, 44 with gate 0.60, 36 at 90% KV.

**Kaggle estimate with run 3's E128 records** (E042 `scripts/multi_sim.py --multi <run 3>`: 240 tasks drawn from the
eval set, ~335 pairs, KV 1.556M / ratio 0.83, s 1.04, 80 reps): per pair × 32/36/40/44: 65.7/69.6/71.3/71.8 (no
thrash), 65.1/66.5/63.4/59.6 (thrash × 0.6 above 0.88); multi-test + budget 0.5: 74.1/74.6/74.8/75.0 and
70.2/67.8/65.2/62.8. Multi-test finishes nearly every attempt 0 even at 32–36 workers, so the robust choice is the
lowest worker count that keeps throughput (36, or 32 for margin), not 44+.

## Context guard and the frozen-input candidate (2026-10-26 ~02:20 UTC)

- Run 3's d35bdbdc (3 test inputs) failed both attempts with vLLM 400s: prompt 108–135k + max_tokens > max-model-len
  163,840. Per-pair runs never hit it (0 in E043's 334 records, the Kaggle bench, the pods). New opt-in
  `--context-limit N` (token-exact path): each turn's max_tokens is capped to N − prompt − 16; with < 1024 tokens of
  room the attempt ends as stop="context" through the session-answer / fallback-solve path. Solver: ARC_CONTEXT_LIMIT.
  Unit-tested (fake turn: stop "context", session answer recovered); real check on d35bdbdc queued on box 1.
- `kaggle/dsv41_tools/build_frozen_mt.py`: Version 5's notebook and exact inputs + settings cell + cell 0b with the
  pinned solver / harness (sha256-checked; fallback to v3 per-pair at 36 workers). Pinned: harness 77be7dfb (= pods'
  ad01d2e3 + the default-identical `--multi-prompt` + the context guard), solver 4b344011 (30522332 + ARC_CONTEXT_LIMIT).
- dsv41-tools-submit-mtf Version 2 = 36 workers, ARC_MULTI_TEST=1, ARC_MULTI_BUDGET=0.5, ARC_CONTEXT_LIMIT=163840,
  smoke commit pushed 02:20 UTC. Worker count is final only after the pods' 3+ h KV results.

**Run 3** (`e043_e128t_mt3_sub80_tx_b05`, 00:28-02:20 UTC): run 2 plus `--multi-budget 0.5` (a task with n test
inputs gets 1 + 0.5 (n − 1) times the output budget, soft budget and turn cap: 120k / 60 turns for n = 2, 160k / 80
for n = 3).

| E128, 45 tasks, 2 attempts (89 pairs: d35bdbdc's errored attempts excluded) | pass@1 | pass@2 | tokens per task-attempt | turns |
|---|---|---|---|---|
| per-pair attempts (baseline) | 69.1% | 77.5% | 77.1k | 36.3 |
| run 2, one rollout per task, 80k | 58.4% | 68.5% | 46.2k | 19.3 |
| **run 3, one rollout per task, `--multi-budget 0.5`** | **71.3%** | **83.1%** | **54.5k (−29%)** | **23.8 (−34%)** |

- Paired pass@2 vs the baseline: both 63, per-pair only 6, multi only 11, neither 9. Counting d35bdbdc's two
  errored attempts as wrong (all 92 pairs), per-pair attempt accuracy is 69.0%, equal to the baseline. Task-weighted
  (harness log): pass@1 70.0%, pass@2 81.1%.
- No-answer 9/90 (run 2: 24): 6 at the output budget, 1 at the turn cap, 2 errors. The errors are both attempts of
  d35bdbdc (3 test inputs, 160k budget): the prompt reached 108k / 135k tokens and prompt + max_tokens passed
  max-model-len 163,840 (HTTP 400). Main then added `--context-limit N` (validated below).
- Another harness bug: in 4e34c42c the model defined its own `eval(grid, H, bg)` in the session, and the submit cell
  calls the bare `eval(...)`, so the session-answer and fallback-solve paths both failed. The cell should call
  `builtins.eval` / `exec` / `compile`.

**Pods at 03:00 UTC** (Kaggle KV 1.557M unless noted; pod sub-agent's timeline in the scratchpad pods_timeline.md):
no live run thrashing, 0 preemptions, memory flat (103–114 GiB). KV at ~0.8 h: per pair × 36 0.37, multi + b0.5 × 36
0.42, × 40 0.46, multi × 44 (80k) 0.46, × 44 + b0.5 0.49, × 52 + gate 0.54. Per pair × 36 (Version 5) plateaus at
0.57–0.66 from 1.8 h to 3.5 h. Multi + b0.5 × 36 is still rising at 1.2 h (0.545, +0.33/h, like × 44 without budget
before its 2 h thrash): the budget-scaled attempts hold more KV each. Added multi + b0.5 × 32 (pod 4, ~03:05).
Pair accuracy of finished pod attempts matches E043 on the same pairs (multi 85% vs 85%; per pair 70.9% vs 78.3%
on 86 pairs, within noise).

**Run 4** (`e043_e128t_mt3b_sub80_tx_b05`, 01:44-03:11 UTC): fresh attempts 0/1 of run 3's configuration (new salt).
pass@1 69.5%, pass@2 83.9% on the 87 pairs without errors (baseline on all 92: 69.0% / 78.3%), 53.2k tokens and 23.5
turns per task-attempt; task-weighted 67.8% / 83.0%; errors counted as wrong, per-pair attempt accuracy 67.4% vs 69.0%.
- Paired pass@2 vs the baseline: both 63, per-pair only 4, multi only 10, neither 10. Against run 3 (same config):
  7 vs 8 discordant pairs, so run 3 replicates.
- No-answer 6/90: 3 at the budget without `answer` or `solve`, 1 at the turn cap (its fallback `solve` raised), 2
  context-length errors, now also on an n = 2 task (4e34c42c: prompt 104.9k + max_tokens 64k > 163,840).
  `--fallback-solve` saved 1 budget-stopped attempt and the session `answer` another.

**Run 5** (run 3's tag, `--attempts 4 --only-attempts 2,3`, 02:25-03:53 UTC, 0 errors): attempts 2/3 of the same
configuration. pass@1 66.8%, pass@2 80.4% (92 pairs), 56.5k tokens and 23.9 turns per task-attempt; paired pass@2
vs the baseline: both 66, per-pair only 6, multi only 8, neither 12. No-answer 8/90 (7 budget, 1 turn cap; none had
`answer` or `solve`); the fallbacks saved 3 more (fallback-solve 2, session `answer` 1).

**Pooled `--multi-budget 0.5`** (`scripts/e128_pooled.py`: run 3 attempts 0-3 + run 4 attempts 0-1, 6 attempts per
pair, 92 pairs; the 4 context-length errors count as wrong):

| | pass@1 | pass@2 | tokens per task-attempt | turns |
|---|---|---|---|---|
| per-pair attempts (baseline, 2 attempts) | 69.0% | 78.3% | 77.8k | 36.4 |
| **one rollout per task, `--multi-budget 0.5`** | **67.8%** | **81.2%** (disjoint pairs 80.4 / 80.4 / 82.6; unbiased 81.5%) | ~55k (−29%) | ~24 (−35%) |

With `--multi-budget 0.5`, one rollout per task ties per-pair E128 on pass@1 and is ~3 pp better on pass@2, at about
70% of the tokens and two thirds of the turns per task-attempt.

**`--context-limit 163840` check** (main's guard: each turn's max_tokens ≤ limit − prompt − 16; with < 1024 tokens
of room the attempt ends as `stop="context"` and takes the session fallbacks). On d35bdbdc (3 test inputs; 3
context-length errors in runs 3-4), 4 attempts with the guard: no errors, all submitted (pairs right 3/3, 1/3, 1/3,
3/3; 14-41 turns; up to 101k prompt). In 2 turns the guard lowered max_tokens, where a 400 would have come before; no
attempt reached `stop="context"`. A small-limit test (`--context-limit 16000`, 2 attempts) ended both with
`stop="context"` and no error: one was answered by the fallback `solve`, the other had nothing to fall back on.

**Run 6** (`e043_e128t_mt6_sub80_tx_b025_ctx`, 03:18-04:19 UTC, harness sha256 7c936ed3, with main's builtins fix):
run 3's flags with `--multi-budget 0.25 --context-limit 163840` (100k / 50 turns for n = 2, 120k / 60 for n = 3).

| E128, 45 tasks / 92 pairs | pass@1 | pass@2 | tokens per task-attempt | turns | no answer |
|---|---|---|---|---|---|
| per-pair attempts (baseline) | 69.0% | 78.3% | 77.8k | 36.4 | |
| `--multi-test`, 80k (runs 1-2) | 58.7-60.3% | 69.6-76.1% | 46.5-47.8k | 19.3-21.4 | 24/90 |
| `--multi-budget 0.5` (runs 3-5 pooled, 6 attempts) | 67.8% | 81.2% | ~55k | ~24 | 6-9/90 |
| `--multi-budget 0.25` + context guard (run 6) | 67.4% | 80.4% | 52.7k | 21.9 | 12/90 |

- Run 6, paired pass@2 vs the baseline: both 67, per-pair only 5, multi only 7, neither 13. Task-weighted 68.5% / 80.0%.
- 0.25 is in the noise band of 0.5 (pass@2 −0.8 pp vs the pooled value) at ~5% fewer tokens, with more budget
  stops: 11 at the budget and 1 at the turn cap, none with `answer` or `solve`; the fallbacks saved 3 more. No attempt
  reached `stop="context"`, and there were no errors.
- Across all runs every no-answer attempt but one had neither `answer` nor `solve` in the session, so
  `--fallback-solve` rarely helps (4 saves in 450 attempts). The losses are attempts that are still exploring when the
  budget ends.

**Run 7: the final harness on all 120 eval tasks** (`e043_e128t_final_all`, 04:20-05:52 UTC, harness sha256
7c936ed3, 160 workers, 0 errors): `--rescue-code --attempts 2 --multi-test --multi-budget 0.5 --token-exact
--fallback-solve --context-limit 163840`. Baseline: per-pair E128 on the same 120 tasks (E043).

| E128, 120 tasks, 2 attempts | pass@1 (task-weighted) | pass@2 (task-weighted) | pass@1 / pass@2 (167 pairs) | tokens per task-attempt | turns |
|---|---|---|---|---|---|
| per-pair attempts (E043 baseline) | 72.8% | 82.5% | 71.9% / 81.4% | 49.0k | 23.7 |
| **final harness** | **73.7%** | **84.6%** | **74.0% / 85.6%** | **40.8k (−17%)** | **18.8** |
| ↳ 75 single-test tasks (75 pairs): baseline → final | | | 75.3% / 85.3% → 73.3% / 82.7% | 31.8k → 32.4k | 16.2 → 16.7 |
| ↳ 45 multi-test tasks (92 pairs): baseline → final | | | 69.0% / 78.3% → 74.5% / 88.0% | 77.8k → 54.8k | 36.4 → 22.2 |

- Paired pass@2 over 167 pairs: both 130, final only 13, per-pair only 6, neither 18. Single-test tasks 3 vs 5
  (`--multi-test` does not change them; token-exact, the fallbacks and sampling noise), multi-test tasks 10 vs 1.
- The harness log prints task-weighted 72.5% / 84.6% for run 7 (its pass@1 is computed differently); the table uses one
  computation for both files.
- No-answer 13/240: 9 at the output budget, 3 at the turn cap, 1 `stop="context"` (20a9e565, n = 2: 59 turns, 118k
  tokens, the next prompt would have been 163,842 tokens). All but one had no `answer` or `solve`. The exception,
  800d221b (single test), had a `solve` whose value failed in `json.dumps`: the single-test cell still converts only a
  top-level array, the same numpy-list issue as the old multi bug, left unchanged on purpose. The fallbacks saved 3
  more (session `answer` 2, `solve` 1). 2 rescues; `token_exact` over 4502 turns: 1283 breaks, 5 resyncs, 4 parse
  fallbacks.
- The server was saturated for the first 40 min (128 running + ~30 queued, ~2.4k gen tok/s). The running count always
  matched the attempts in flight, so this tunnel did not stall (box 2 saw `--token-exact` uploads stall a 185 ms tunnel).

**Server over all runs** (23:14-05:53 UTC, up to 3 runs at once): 727M prompt tokens at 97.9% prefix-cache hits, 38.2M
generated tokens (~1.6k tok/s over 6.6 h; ~1.2-1.4k tok/s with one 90-stream run, ~2.4k at 128 streams), KV peak 7.8%
of 20.67M tokens, at most 128 running and 70 waiting, 0 preemptions. Per stream ~12-20 tok/s, so an 80k-token attempt
takes over an hour. KV is not the limit on this box: concurrency is (`--max-num-seqs 128`).

Results (`evals/results/`, prefix `tools_persistent_grid_lean2sub_dsv41_low_e043_`): `e128t_mt_sub80` (run 1),
`e128t_mt2_sub80_tx` (2), `e128t_mt3_sub80_tx_b05` (3 = attempts 0-1, 5 = attempts 2-3), `e128t_mt3b_sub80_tx_b05`
(4), `e128t_mt6_sub80_tx_b025_ctx` (6), `e128t_mt3c_ctx` and `e128t_ctxsmall_test` (context-guard checks), `e128t_final_all` (7). Summaries:
`scripts/e128_run_summary.py <run> <baseline> [<other run>]`, `scripts/e128_pooled.py`, and
`scripts/remap_attempts.py` (attempts 2/3 as 0/1 for `multi_test_compare.py`).

## Decision (2026-10-26 ~04:50 UTC): next submission = dsv41-tools-submit-mtf Version 4

Settings: multi-test + `ARC_MULTI_BUDGET=0.5` + KV gate `ARC_KV_ADMIT_MAX=0.6` (min 16 running), up to 44 workers,
`ARC_CONTEXT_LIMIT=163840`, Version 5's server settings and exact inputs; pinned harness 7c936ed3 (context guard +
builtins fix), solver 4b344011. Smoke commit passed (04:39 UTC; KV 1,554,183; 6/6 submitted).
Evidence:
- Accuracy (E128, 45 multi-pair tasks): pooled 6 samples pass@1 67.8% / pass@2 81.5% vs per pair 69.0% / 78.3% (E043)
  and 63.6% / 76.1% (Kaggle flags, box 2 run A); −29% tokens, −35% turns. Budget 0.25: within noise, −5% tokens only.
- KV (pods at Kaggle KV): gate 0.6 × 44 held KV 0.61–0.72 from 1.2 h to 2.5 h (no thrash, 0 preemptions), running fell
  44 → 21–26 and tok/s 340 → ~230, yet attempts/h (21.5) equal ungated × 36 (21.7). Ungated × 36 peaked at KV 0.834
  at 2.48 h (no thrash, 0.05 from the edge); × 44 ungated thrashed at 2.0 h; × 52 with gate 0.75 thrashed at 1.65 h.
- Context guard validated on d35bdbdc (4/4 attempts, 0 errors; previously 3 × 400s). Pods without it hit one more
  context error (b6f77b65, n = 3).

**Revision 05:36 UTC: candidate = mtf Version 5 (max 36 workers, gate 0.6).** The 90%-KV stress test of Version 4's
config (gate 0.6 × 44 at 1.403M) crossed KV 0.80 at 1.36 h with 41–43 running: the gate stopped admissions at
0.95 h, but the running set's contexts grew faster than finishes freed them (as with gate 0.75 × 52). Ungated × 36
at 90% KV thrashed at 3.41 h (pod 5). Fewer max streams lowers the running set's peak (monotone), and the simulator
puts gated multi at × 36 within 0.2 of × 44. Version 5 = Version 4 with ARC_WORKERS 36; stress test
mt36_b05_gate60_kv90 on pod 2, primary run mt36_b05_gate60 on pod 6 (since 04:39).

**Per-pair E128 with Kaggle flags, 4 samples (box 2 runs A + B; B stopped at 157/184 by decision, 27 tunnel-stall
errors):** same (task, pair, attempt) cells (335): per pair 67.5% vs one rollout per task (run 3, attempts 0–3) 70.7%;
on the 67 pairs with all 4 + 4 attempts: pass@1 73.5% vs 75.4%, mean pass@2 83.6% vs 87.3%; tokens per task-attempt
~74.6k vs 55.5k, turns 34.3 vs 23.8. Multi-test is a tie or slightly better at ~25–35% fewer tokens. Lesson for
remote boxes: `--token-exact` uploads ~1–2 MB per turn; a single lossy SSH tunnel throttles the run (use several
`ssh -C` tunnels with a comma-separated `--base-url`, or run the harness on the box).

## Final build (2026-10-26 06:27 UTC)

- **dsv41-tools-submit-mtf Version 6** = Version 5's notebook and exact inputs + settings cell (ARC_WORKERS 36,
  ARC_MULTI_TEST 1, ARC_MULTI_BUDGET 0.5, ARC_CONTEXT_LIMIT 163840, ARC_KV_ADMIT_MAX 0.6 / min 16, max-num-seqs 64,
  util 0.945, Humming dense, graphs ≤ 64) + cell 0b with harness a8a5b665 (7c936ed3 + nested-array conversion in
  single-test answers too) and solver 4b344011. Cell 0b checks hashes, required features and the settings; in the
  commit run a failure raises (log "E046 CHECK FAILED"), in the competition rerun it restores v3's kit files and
  runs per pair at 36 workers ("E046 FALLBACK"). Smoke commit pushed 06:27.
- **chankhavu/dsv41-tools-mtf-bench Version 1** = the same cells after a bench cell: the 120 eval tasks with
  solutions, stop-new 2.75 h, force 3.0 h, hard 3.08 h, kill 3.25 h (Kaggle stress test, ~3.3 h of GPU).

## Pod tooling and timelines (`pods/`)

- `podprep.sh` (kit + its vLLM overlay over the box_install venv, tool venv, eval data), `podrun.sh <name> [ENV=...]`
  (the Kaggle notebook cells via `local_run.py` against the pod venv), `memlog.sh` (host memory per minute),
  `pod4_setup.sh` (wait for a new Runpod pod via the API, then install + E128 download (HF token via stdin) + prep +
  run). The pod fleet sub-agent's watcher and analysis helpers are in `pods/watch/` (`bigprefill.py` scans traces
  for full re-prefills).
- `pods/pods_timeline.md`: the fleet timeline and summaries; `pods/runs/hist_<run>.txt`: per-run metric histories.
- `heavy60_ids.txt` (60 largest eval tasks, the heavy-phase stress test), `multi_ids.txt` (45 multi-pair tasks).
- Summary of all inference findings: `notes/2026-10-26_dsv41_kaggle_inference_findings.md`.
