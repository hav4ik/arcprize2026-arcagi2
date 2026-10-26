# E050: reasoning effort 25 vs 50 on E128 (dsv41-tools-submit-mtf V6 pipeline)

**Question.** Does `reasoning_effort` 25 (instead of the submitted 50) cut output tokens per task by >= 10-15% while
pass@1 stays within 2-3 pp? The deadline simulator says -15% tokens at equal pass@1, with thrash removed, would lift
the projected Kaggle score from ~73.5 to ~77.

**Verdict: no. Effort 25 does not cut tokens.** Effort 25 / effort 50 output tokens per task: **1.029** (sum
ratio, 95% bootstrap CI [0.952, 1.111]; geometric mean 1.021). The CI excludes a 10% cut. Task pass@1 is a tie: 72.4%
vs 71.4% (+1.0 pp, CI [-5.3, +7.2], task wins 11/10). Keep effort 50: the simulator's lift does not materialize.

## Setup (2026-10-26)

- Two Vast 4x RTX 4090 boxes, one arm each, run at the same time:
  - effort 25: box 2 (57817226), 13:34-16:27 UTC;
  - effort 50: box 1 (57816747), 14:19-17:04 UTC.
- Pipeline: `podrun.sh` running `local_run.py` (the notebook cells; sha256 e792c94b) with the pinned V6 code
  (`kaggle/dsv41_tools/pinned_mtf_v6/`):
  - harness `run_deepseek_tools.py` a8a5b665;
  - solver 4b344011 plus one edit, giving fc8ba4d1 (`runs/*/solver_dsv41_e050.py`). In `EXTRA_BODY`,
    `"reasoning_effort": 50` became `int(E("ARC_EFFORT", "50"))`.
  - The other kit files match the kit MANIFEST.
- V6 settings, the same in both arms (`runs/*/env.txt`):
  - `ARC_WORKERS=36`, `ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, `ARC_CONTEXT_LIMIT=163840`;
  - `ARC_KV_ADMIT_MAX=0.6`, `ARC_KV_ADMIT_MIN_WORKERS=16`, `ARC_MAX_NUM_SEQS=64`, `ARC_GPU_UTIL=0.92`;
  - humming + cudagraph cap 64;
  - `ARC_ATTEMPTS=1`, all 120 tasks of the Kaggle eval json (same md5 as `data/`), per-arm cache salt;
  - deadlines pushed out of the way: stop-new 30 h, force 31 h, hard stop 32 h, kill switch 33 h.
- The vLLM command lines are identical across arms (`runs/*/dsv41_logs/server_cmd.json`). KV is 2.12M tokens per box
  at util 0.92; Kaggle's 4xL4 has 1.554M.
- Effort reached the prompt: the server's `/tokenize` renders "Reasoning Effort: 25 (range 1-100, ...)" and "...: 50"
  respectively. The harness's minimal-effort header ("Reasoning Effort: Minimal. Keep your thinking short ...") is a
  system message right after that prefix, in both arms.
- The JIT caches were symlinked to each box's smoke-test cache. This does not change behaviour.
- Server health was the same on both boxes:
  - 519 / 537 generated tok/s over the run;
  - prefix hits 0.972, KV peak 0.61 / 0.62 (the 0.6 gate engaged), 0 preemptions, 0 errors, 0 loop cuts.

## Results (`analyze.py`, `results.json`)

| arm | tasks | pass@1 (task) | pass@1 (pairs) | out tok/task mean | median | p90 | turns | no answer | train_pass | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| effort 25 | 120 | **72.4%** | 125/172 = 72.7% | **43.4k** | 37.3k | 91.7k | 20.3 | 8 (6.7%) | 84 | 2.89 h |
| effort 50 | 120 | **71.4%** | 120/172 = 69.8% | **42.2k** | 32.8k | 82.5k | 19.2 | 7 (5.8%) | 80 | 2.75 h |
| run 7 attempt 0 (effort 50, 8x box) * | 114 | 72.8% | 117/159 = 73.6% | 40.7k | 34.3k | 80.0k | 17.6 | 5 | 82 | |
| run 7 attempt 1 * | 114 | 75.3% | 118/159 = 74.2% | 39.2k | 31.0k | 85.0k | 19.4 | 6 | 76 | |
| effort 25 on the 114 tasks * | 114 | 73.7% | 119/159 = 74.8% | 41.4k | 32.2k | 84.3k | 19.7 | 8 | 83 | |
| effort 50 on the 114 tasks * | 114 | 72.7% | 114/159 = 71.7% | 40.2k | 32.3k | 80.0k | 18.5 | 7 | 78 | |

\* E046 run 7 (`evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all.jsonl`): the same
harness config at effort 50 (E043's `eval_arc.sh` sends 50), on the GitHub eval set. The rows marked * cover the 114
tasks whose prompt is identical in both sets. Pairs are matched by test-input content.

- The Kaggle eval json has 5 extra test pairs (4a21e3da, abc82100, b6f77b65, f560132c, faa9f03d).
- d8e07eb2 and faa9f03d have different train pairs.
- `analyze.py` reproduces run 7's published 73.7% / 74.0% / 40.8k / 18.8 turns exactly.

Stops: effort 25 submitted 111, output budget 6, max turns 3; effort 50 submitted 109, output budget 7, max turns 4.

Single- vs multi-test tasks:

| arm | 71 single-test tasks: pass@1, tokens | 49 multi-test tasks: task pass@1, tokens |
|---|---|---|
| effort 25 | 70.4%, 33.0k | 75.2%, 58.4k |
| effort 50 | 74.6%, 32.6k | 66.7%, 56.0k |

The single/multi split moves in opposite directions; this is sampling noise.

### Paired by task (a vs b)

| comparison | tasks | pass@1 a vs b | diff [95% CI] | task wins a/b (sign p) | token ratio a/b [95% CI] | geo-mean ratio [95% CI] |
|---|---|---|---|---|---|---|
| **effort 25 vs effort 50** | 120 (172 pairs) | 72.4% vs 71.4% | +1.0 pp [-5.3, +7.2] | 11/10 (1.00) | **1.029 [0.952, 1.111]** | 1.021 [0.934, 1.116] |
| effort 25 vs effort 50, 114 tasks | 114 | 73.7% vs 72.7% | +1.0 pp [-5.6, +7.6] | 11/10 (1.00) | 1.029 [0.946, 1.120] | 1.018 [0.927, 1.120] |
| effort 25 vs run 7 attempt 0 | 114 | 73.7% vs 72.8% | +0.9 pp [-6.6, +8.3] | 11/10 (1.00) | 1.016 [0.929, 1.110] | 1.008 [0.916, 1.108] |
| effort 25 vs run 7 attempt 1 | 114 | 73.7% vs 75.3% | -1.6 pp [-9.8, +6.7] | 15/16 (1.00) | 1.054 [0.966, 1.156] | 1.064 [0.972, 1.167] |
| effort 50 vs run 7 attempt 0 | 114 | 72.7% vs 72.8% | -0.1 pp [-8.3, +7.9] | 13/15 (0.85) | 0.987 [0.896, 1.084] | 0.990 [0.892, 1.093] |
| effort 50 vs run 7 attempt 1 | 114 | 72.7% vs 75.3% | -2.6 pp [-10.1, +4.8] | 10/13 (0.68) | 1.024 [0.935, 1.125] | 1.045 [0.950, 1.152] |
| run 7 attempt 0 vs attempt 1 (same config: noise) | 114 | 72.8% vs 75.3% | -2.5 pp [-11.0, +5.8] | 15/16 (1.00) | 1.038 [0.942, 1.149] | 1.056 [0.951, 1.177] |

- **The effort-50 arm reproduces run 7.** Token ratio 0.99 / 1.02, pass@1 within run 7's own attempt-to-attempt
  spread. So the 4090 boxes and the V6 pipeline behave like the 8x box with the E046 harness.
- **Effort 25 against run 7 looks the same: no token change.** Its ratio to every effort-50 sample (1.016-1.054) is
  inside the noise of two effort-50 samples against each other (1.038).
- **Noise.** One paired run on 120 tasks gives a pass@1 difference CI of about ±6 pp. Two attempts of the same
  configuration differ by 2.5 pp. The "within 2-3 pp" criterion can only be judged on the point estimate. That is
  moot here, because the token criterion fails clearly.

### Where the tokens go (traces, attempt 0)

| arm | turn-1 out tokens mean / median | out tokens per turn mean / median / p90 | turns | thinking share (chars) |
|---|---|---|---|---|
| effort 25 | 413 / 172 | 2138 / 936 / 5318 | 20.3 | 0.87 |
| effort 50 | 542 / 179 | 2199 / 967 / 5369 | 19.2 | 0.87 |

- Effort 25 shortens only the first turn's mean (-24%). That turn is about 1% of an attempt's output.
- Later turns are the same length (-3% per turn), and attempts take about 1 more turn, which cancels it.
- The thinking share is unchanged.
- A likely reason: the minimal-effort header and the tool loop already set how terse the model is, so the numeric
  budget in the first system block barely registers on E128.

## Files

- `analyze.py`: the whole analysis. With no arguments it reads `runs/eff25` and `runs/eff50`, prints the tables
  above and writes `results.json`.
- `runs/eff25/`, `runs/eff50/`: each arm's `out/`:
  - `submission.json`, `dsv41_traces.tar.gz`;
  - `dsv41_logs/`: attempts.jsonl, solver.log, metrics.jsonl, server.log, server_cmd.json, status.json, stage.log;
  - plus `run.log`, `env.txt`, `launched_utc` and the edited solver.
- **Secret note:** the solver dumps env keys starting with `HF_` into `server_cmd.json`, and the Vast image sets
  `HF_TOKEN`. Its value is redacted in both copies here (`"<redacted>"`). Copies from other Vast or Runpod boxes need
  the same check before syncing.
