# E004: Python tool use for DeepSeek V4.1-Flash

## Hypothesis
Flash solves most failures if it can run code.
- **E003:** 25–45% of the tokens in a trace go to building and counting grids by hand.
- **E007:** of the wrong attempts,
  - 32% are execution errors (rule right, hand execution wrong);
  - 45% are partial rules (core right, sub-rule wrong). Most of these fail a training pair.

A persistent Python session should fix the execution errors. A `transform()` that is checked on every training pair before it is accepted (`code` mode) should also catch most partial rules. Expected outcome: equal or better accuracy at fewer output tokens per attempt.

**Baseline (E001d)**, Flash max effort, ARC Prize prompt, no tools:
- 63.1% pass@1, 78.1% pass@2;
- ~53k output tokens per call.

## Harness
- **Files:**
  - `evals/run_deepseek_tools.py`: the agent loop, scoring and resume;
  - `evals/sandbox_client.py`: the sandbox HTTP client;
  - `evals/tests/test_deepseek_tools.py`: 16 tests.
- **Spec:** `notes/specs/tool_use_harness_spec.md`.
- **Prompt:** ARC Prize's prompt verbatim. It is scored like E001d, with the same parser and task-weighted pass@k.
- **Sandbox session:** each attempt owns one persistent IPython session in `evals/sandbox_docker/start.sh`, which has no internet and no host mounts.
  - The session is preloaded with `train`, `test_input` and `np`. The test answers never enter it.
  - A cell that times out keeps its state and is reported as `[timeout]`.
- **Modes:**
  - `grid`: tool `run_python`; the final answer is a JSON grid in text, parsed as in E001d.
  - `code`: `run_python` plus `submit_solution(code)`.
    - The harness runs `transform` on every training pair. Any failure returns the predicted and expected grids so the model can revise.
    - If every pair passes, it applies `transform` to `test_input` and the attempt ends.
    - If the model never gets a submission accepted, its last text grid is used.
- **Budgets per attempt:**
  - 200k output tokens in total, with 8k held back for a forced "Finalize now" turn;
  - at most 40 turns;
  - 30s per cell.
- **No automatic API retries.** Failed attempts are recorded with `error` and re-run on resume. Each re-run is billed again.
- **Records:** `evals/results/tools_persistent_<mode>_<model>_<effort>[_tag].jsonl`. Full message traces, including reasoning and every cell, are saved as `.messages.json.gz` in the matching folder. `evals/report.py` reads these files.
- **Cost guard:** the harness refuses to launch if the worst-case output cost is over $10, unless `--allow-expensive-run` is passed. `--price-multiplier` defaults to 2 (peak pricing). Use `1` off-peak; the National Day holiday is off-peak through 2026-10-07.

## Commands
```bash
evals/sandbox_docker/start.sh            # prints e.g. http://172.18.0.2:6000
export ARC_SANDBOX_URL=http://172.18.0.2:6000
ARC_SANDBOX_URL=$ARC_SANDBOX_URL .venv/bin/python -m unittest discover -s evals/tests   # 16 tests incl. Docker
cd evals
../.venv/bin/python run_deepseek_tools.py --mode code --attempts 1 --tasks <ids> --tag pilot --price-multiplier 1 --dry-run
```
Drop `--dry-run` to launch. Re-running the same command resumes.

## Plan (Flash only, staged, each stage needs approval)

| Stage | What | Calls | Est. output cost (off-peak) | Worst case |
|---|---|---|---|---|
| E004a pilot | `code` mode, 1 attempt, 20 pairs / 15 tasks (below) | 20 | ~$0.8 | $2.4 |
| E004b | `code` mode, 1 attempt, all 167 pairs | 167 | ~$6.5 | $20 (needs `--allow-expensive-run`) |
| E004c | 2nd attempt only where attempt 1 had no accepted submission | ≤167 | ≤$6.5 | — |
| (optional) | `grid` mode on the pilot set, to separate the effect of the tool from the effect of train verification | 20 | ~$0.8 | $2.4 |

Input tokens are extra and grow with the number of turns. Most of them are cache hits ($0.003/M); cache misses are $0.15/M. The estimates use 65k output tokens per attempt; the pilot will measure the real figure.

**Pilot set** (`--tasks 13e47133,6e4f6532,aa4ec2a5,dd6b8c4b,e12f9a14,2b83f449,dfadab01,e87109e9,31f7f899,3dc255db,a251c730,2ba387bc,7b5033c1,136b0064,3e6067c3`):

| Group | Tasks | Why |
|---|---|---|
| E001d unsolved, labelled execution in E007 | 13e47133 (2 pairs), 6e4f6532 (2), aa4ec2a5, dd6b8c4b (2), e12f9a14 (2) | Code should fix these |
| E001d unsolved, labelled partial rule; a training pair contradicts the model's rule | 2b83f449, dfadab01, e87109e9 | The train check should flag these |
| One of two E001d attempts correct | 31f7f899, 3dc255db, a251c730 | Stability |
| Easy, both attempts correct | 2ba387bc, 7b5033c1, 136b0064, 3e6067c3 (2) | Regression check and token saving |

E001d on these 20 pairs: attempt 1 correct on 10/20, either attempt correct on 10/20, mean 55.7k output tokens per call. The comparison is per pair, not task-weighted.

**Go/no-go for E004b:** the pilot is at least 13/20 with mean output tokens at most ~55k, or it shows a clear win on the execution group without regressions.

## Results (2026-10-03, in progress)

### Harness fixes made during the runs
- **v1 → v2:** `submit_solution` checked against the session's `train`/`test_input`, which the model sometimes rebinds by accident (e.g. `train = []`).
  - This made one check vacuous and wrongly rejected two others.
  - v2 embedded host copies of `train`/`test_input`.
- **v2 → v3** (recommended by the safety audit): submitted code now runs on the inputs only, in a fresh session with no outputs in memory, and the host compares the results.
- **Concurrency:** the sandbox HTTP timeout is now `2×timeout+60` (24 shared single-process workers).
- **New options:** `--pairs`, and `--prompt {plain,lean,lean2}`, which also appears in the output filename.
- **Balance check:** `evals/balance_log.py` logs the balance every 10 min next to recorded cost estimates (`evals/results/balance_log.tsv`). Billing matched off-peak prices: a $18.47 drop against $18.78 estimated.

### Pilot (20 pairs / 15 tasks, hard-biased; E001d attempt 0 = 10/20 at 58k tokens)

| Config | Correct | Mean / median output tokens |
|---|---|---|
| grid, plain, max (v1) | 13/20 | 104k / 91k |
| code, plain, max (v1, buggy check) | 12/20 | 97k / 86k |
| code, plain, max (v2) | 12/20 | 104k / 94k |
| grid, lean, max | 15/20 | 74k / 36k |
| code, lean, max | 15/20 | 91k / 74k |
| grid, lean, high | 14/20 | 57k / 30k |
| code, lean, high | 12/20 | 76k / 51k |

- With the plain prompt the model solves by hand first: its first tool call comes after a median of 48–86k tokens.
- The lean prompt (compute with code, never by hand) moves the first call to 2–15k tokens.
- Several attempts used the whole 200k budget in one or two turns. The 8k final turn never produced a grid.

### Effort sweep, full public eval (attempt 0, lean2, 120k total / 64k per turn / 24k final)

| Config | pass@1 (ARC Prize, task-weighted) | Output tokens / attempt | Est. cost |
|---|---|---|---|
| E001d attempt 0, no tools | 63.1% | 51.9k (successful calls; 10 failed calls not counted) | — |
| grid, lean2, **low** | **66.7%** | **49.9k** | $5.7 |
| grid, lean2, high | 67.1% | 54.8k | ~$6 |
| grid, lean2, max | 65.3% | 65.0k | $7.2 |
| code, lean2, high | 61.9% | 57.8k | $6.7 |
| code, lean2, max | 60.7% | 65.2k | $7.2 |

**Grid at low effort beats E001d on both accuracy and output tokens per attempt.**
- Simulating a 100k cap on the same records keeps 66.7% at 48.6k tokens. A prospective cap-100k run is in progress (`_cap100`).
- With tools, max effort is the most precise per answer, but it leaves twice as many pairs unanswered: it runs fewer, longer experiments and hits the cap.
- Code mode trails grid mode because it has no best-guess exit. `submit_solution` rejected nothing: the model only submits after checking on train.

Trace reviews: `trace_review_behaviour.md` (pilot), `trace_review_behaviour_full.md`, `trace_review_safety.md` (pilot), `trace_review_safety_full.md`.

**Safety:** nothing concerning in 870 full-run traces (10,159 code cells). Two questionable cases:
- After wiping `train`, the model listed `/tmp`, `/root` and `/home` and saw other runs' scratch files. It did not open them.
- The model tried to recall the original task's answer and listed real ARC-AGI-2 eval task ids, which is further evidence of contamination. It recalled no solution.

**Open sandbox issues:**
- `/tmp` is shared across sessions.
- The container runs as root with no limits.
- Finished sessions' processes are not cleaned up. Around 07:50 UTC about 1,650 leftover processes used ~54 GB of host RAM and caused API and sandbox errors.
  - Mitigation used: stop the runs, restart the container, resume with 20 workers per run.
  - The client's session cleanup needs a fix, to be reviewed by the user.

### Full public eval (120 tasks / 167 pairs, 1 attempt, lean2 prompt, 120k total / 64k per turn / 24k final)

| Config | pass@1 | Output tokens per attempt | Cost |
|---|---|---|---|
| E001d attempt 0 (no tools) | 63.1% | ~53k | — |
| grid, lean2, max | **65.3%** | 65k | $7.2 |
| code, lean2, max | 60.7% | 65k | $7.2 |

**Code-mode submissions:**
- 109 of 167 were accepted, and 100 of those (92%) were correct.
- Attempts with no accepted submission are almost never right: 0/58.

**Combining runs into one answer per pair** (`evals/candidate_pass1.py`: accepted submissions first, then a majority vote, then the E001d attempt-0 fallback):
- code + E001d fallback: **75.7% pass@1**;
- code + grid + E001d fallback: 76.9% pass@1, 83.1% pass@2.

These cost more than one attempt per pair: 73–88k tokens per pair when cascaded.

**Tokens:**
- About 50 attempts per run end without any answer, about half by hitting the cap. The final turn often still writes a tool call as plain text.
- Taking the last grid from those traces recovers only 2/108. They are genuinely unsolved, so they only waste tokens.
- Lowering the cap doesn't help. Under max effort even correct attempts are long, so cutting the cap to 80k gives ~56% at 55k tokens.
- The remaining lever is reasoning effort: high and low effort runs are in progress.

## History
- 2026-10-02: a pilot on 2 easy tasks used the old stateless sandbox, which had no filesystem isolation. It solved 4/4 attempts with 2–6 tool calls each, at ~62k tokens per call (`evals/results/tools_deepseek-flash_max_pilot.jsonl`). That harness was then replaced by the persistent one above.

## Final result (2026-10-03 09:30 UTC): candidate E004-GL

**Candidate:** grid mode, `lean2` prompt, `low` effort, 120k total / 64k per turn / 24k final.
```bash
cd evals && ../.venv/bin/python run_deepseek_tools.py --mode grid --prompt lean2 --effort low \
  --max-total-output 120000 --max-tokens 64000 --final-tokens 24576 --price-multiplier 1 --attempts 1
```

**Comparison** (official public eval, 120 tasks / 167 pairs, ARC Prize task-weighted scoring, all calls completed with no errors):

| System | Samples | pass@1 per sample | Mean pass@1 | Output tokens / attempt | Mean pass@2 (pairs of samples) |
|---|---|---|---|---|---|
| E001d: Flash max effort, no tools | 3 | 65.1, 60.3, 54.2 | 59.9% | 55.7k (median 49.2k) | 74.6% (71.8–79.3) |
| **E004-GL: grid / lean2 / low** | 4 | 66.7, 70.0, 64.4, 62.8 | **66.0%** | **49.5k** (median 41.4k) | 76.4% (75.3–77.8) |
| grid / lean2 / high | 2 | 67.1, 69.4 | 68.3% | 55.2k (median 48.5k) | 78.2% |

**Paired difference in expected pass@1 vs E001d** (per-task success rate averaged over samples; bootstrap over tasks, 10k resamples):
- E004-GL: **+6.1 pp**, 95% CI [+0.3, +12.2], P(≤0) = 0.02, with **11% fewer output tokens per attempt**.
- grid/high: +8.4 pp, 95% CI [+1.9, +14.8], P(≤0) = 0.004, at the same token cost.

**Majority vote over several rollouts (pass@1, more tokens per pair):**
- E004-GL ×3: 75.4%; ×4: 76.9%.
- E001d ×3: 68.6%.

**Notes:**
- E001d was completed on 2026-10-03: its 17 connection-error attempts were retried, and a third sample was added. Its attempt-0/1 pass@2 is now 79.3%, up from 78.1%.
- The pre-retry records are backed up in the session scratchpad only.
- Per-sample pass@1 swings by about ±3–5 pp. Compare means, not single runs. For example, the one cap-100k grid-low sample scored 61.9%.
- **Contamination caveat:** the public eval is in the model's pretraining data; the model even listed real eval task ids. Absolute numbers may be inflated; the comparisons use the same model throughout.

**Spend overnight:** about $77 of the $86.25 balance, including about $2–4 lost to stopped or crashed in-flight calls. Billing matched off-peak list prices throughout.

**Tools added:**
- `evals/candidate_pass1.py`: single-answer ensembles.
- `evals/compare_runs.py`: runs against E001d on the same pairs.
- `evals/sample_stats.py`: per-sample pass@1, mean, sd, pass@2 and voting.
- `evals/balance_log.py`: balance against estimates.

**Next steps:**
1. Fix sandbox session cleanup. Leftover session processes grow without bound; restart the container between big runs until this is fixed.
2. Isolate `/tmp` per session and run the sandbox as a non-root user.
3. Give grid mode a best-answer variable read by the harness, and drop or shorten the forced final turn. The trace review estimates +2.5–3 pp from this.
4. Pair a low-effort and a high-effort attempt for pass@2: 79.2% at about 105k tokens per pair.

## Prompt ablation (2026-10-03/04): plain ARC Prize prompt with tools, full eval, 1 sample each

Same harness as the candidate: grid mode; 120k total, 64k per turn and 24k final token budgets.

| Config | pass@1 | Output tokens / attempt | No answer (cap hit) | Tool calls / attempt | Cost |
|---|---|---|---|---|---|
| grid, **plain**, low | 51.3% | 61.0k | 37 (29) | 9.8 | $6.61 |
| grid, **plain**, max | 47.6% | 86.2k | 75 (68) | 4.6 | $9.06 |
| grid, lean2, low (4 samples) | 66.0% | 49.5k | ~26 | — | — |
| grid, lean2, max | 65.3% | 65.0k | 52 (20) | 10.1 | $7.2 |
| E001d, no tools, max (3 samples, 200k single call) | 59.9% | 55.7k | — | — | — |

**Findings:**
- **With ARC Prize's plain prompt, giving Flash a Python tool lowers accuracy below no tools**, by 9–12 pp, and raises tokens.
  - The model reasons by hand for a long time and calls the tool rarely, 4.6 times per attempt at max effort.
  - It then runs out of budget with no answer: 68/167 attempts at max effort.
- **The `lean2` suffix is worth about 15–18 pp at equal effort and cuts tokens.** Compared to no tools, the gain comes from the prompt together with the tool, not from the tool alone.

**Caveat:** tool runs have a 120k total budget across turns, while E001d had one 200k call. E006 showed E001d loses only 0.4 pp under a 120k cap, but plain-prompt tool runs do hit the cap. A bigger budget would raise their accuracy somewhat, at even higher token cost.

## Sandbox update (2026-10-04)
- `evals/sandbox_docker/start.sh` now sets `NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT` (default 1800 s, override with `SANDBOX_SESSION_TIMEOUT`).
  - This enables the server's built-in reaper: each worker stops its sessions idle longer than the timeout when it handles a request.
  - It addresses the leak where finished sessions' processes accumulated (~1,650 processes, ~54 GB on 2026-10-03).
- **Why 1800 s:** the longest of 2,145 recorded attempts took 15.4 min, so a live attempt is never idle that long.
  - If one ever is, the client re-initialises `train`/`test_input` and tells the model its state was lost.
- **Verification:** the container runs with the setting, and all 16 harness tests pass.
  - The reaping behaviour itself (a session idle past the timeout is stopped) has not been tested end to end here.
- **Limitation:** reaping happens only when a worker receives a request, so idle sessions linger after a run ends. Restart the container between big runs.
