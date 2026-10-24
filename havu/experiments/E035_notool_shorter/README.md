# E035: shorter no-tool reasoning on Flash-1017 (effort level, minimal header, pre-computed stats)

Goal: shorten E034's no-tool traces (pass@2 80.6% at 50.4k output tokens/call) without losing pass@2.
ARC Prize prompt, official 120 tasks / 167 pairs, 2 attempts per arm, 40 workers per arm, run in parallel.

New opt-in options in `evals/run_deepseek_arcprize.py` (defaults reproduce E001d/E034):
- `--effort-header minimal`: a system message in DeepSeek's effort-prefix style; the E030 text minus the code
  sentences ("Commit to the simplest rule that reproduces every training pair, then write the answer").
- `--task-summary`: the E017 host-computed grid summary appended to the prompt, with a no-tool intro ("it is exact,
  so trust it over reading or counting the grids by hand"). E026: without tools DeepSeek spends ~33% of its tokens
  computing these statistics by hand.

DeepSeek's effort levels are themselves prefixes: `high` and `max` add "Absolute maximum…" / "Beyond maximum…"
text, `low` adds none. So the header is tested on `low` only.

| Arm | Flags | tag |
|---|---|---|
| A | max + summary | `arcprize_deepseek-flash_max_e035_sum` |
| B | low | `arcprize_deepseek-flash_low_e035_plain` |
| C | low + summary | `arcprize_deepseek-flash_low_e035_sum` |
| D | low + header + summary | `arcprize_deepseek-flash_low_e035_hdrsum` |
| E | low + header (added at launch) | `arcprize_deepseek-flash_low_e035_hdr` |
| E | low + header | `arcprize_deepseek-flash_low_e035_hdr` |

Baseline: E034 (max, no extras). Approved estimate ~$17–22 off-peak. `launch.sh` waits for 10:05 UTC when started
inside the weekday peak window (a first launch at 06:03 UTC on 2026-10-19 was stopped after 8 s).

## Results
Launched 10:05 UTC 2026-10-19 (off-peak). C, D and E finished at 10:32 with 0 errors (no re-runs were needed).
A and B were stopped early (10:15 and 10:26) because the projected total went over the $32 cap. Both can be
resumed with their `launch.sh` commands. Their metrics over all 120 tasks count missing calls as wrong, so
only the paired subsets below are meaningful for them. Those subsets are the tasks with every call complete,
which skews toward easy and short tasks. All 1,321 completed calls ended with `stop`. Unparsed: 2 (C).
Analysis: `scripts/analyze.py`, with its output in `scripts/analyze_out.txt`.

| Arm | calls | pass@2 | pass@1 a0 | pass@1 mean | pair pass@2 | out tok mean / median / p90 | in tok | cost |
|---|---|---|---|---|---|---|---|---|
| E034 max | 334 | 80.6% | 66.4% | 64.6% | 79.0% | 50.4k / 44.8k / 93.3k | 7.2k | $10.44 |
| C low+sum | 334 | 73.6% | 66.7% | 66.2% | 73.1% | 35.9k / 30.7k / 64.8k | 9.3k | $7.63 |
| D low+hdr+sum | 334 | 75.4% | 67.4% | 62.6% | 74.9% | 31.9k / 28.0k / 57.6k | 9.4k | $6.84 |
| E low+hdr | 334 | 67.9% | 54.6% | 54.2% | 67.1% | 37.6k / 31.2k / 71.3k | 7.3k | $7.87 |
| B low (stopped) | 226 | - | - | - | - | 40.4k / 33.7k / 73.8k | 6.9k | $5.70 |
| A max+sum (stopped) | 93 | - | - | - | - | 45.4k / 40.9k / 83.4k | 8.9k | $2.65 |

The summary adds about 2.1k input tokens per call. Total cost was ~$30.7 at the off-peak list price. Killed
in-flight calls may add a little on top.

Paired deltas (bootstrap 95% CI over tasks complete in both arms):

| Comparison | tasks | d pass@2 | d pass@1 mean | out tokens |
|---|---|---|---|---|
| D - E034 (low+hdr+sum) | 120 | -5.1 [-11.8, +1.2] | -1.9 [-8.1, +4.2] | x0.63 |
| C - E034 (low+sum) | 120 | -6.9 [-14.4, +0.4] | +1.6 [-5.2, +8.5] | x0.71 |
| E - E034 (low+hdr) | 120 | -12.6 [-20.3, -5.1] | -10.4 [-17.0, -3.8] | x0.75 |
| B - E034 (low) | 76 | -9.6 [-18.4, -0.7] | -8.0 [-15.9, +0.0] | x0.83 |
| A - E034 (summary at max) | 28 | +5.4 [-1.8, +14.3] | +11.6 [+1.8, +23.2] | x0.94 |
| C - B (summary at low) | 76 | +5.0 [-3.5, +13.8] | +10.6 [+2.5, +19.0] | x0.86 |
| D - C (header, given summary) | 120 | +1.8 [-4.0, +7.9] | -3.5 [-8.3, +1.5] | x0.89 |
| E - B (header alone) | 76 | -2.6 [-12.5, +6.6] | -3.0 [-12.2, +5.9] | x0.87 |
| D - E (summary, given header) | 120 | +7.5 [-0.4, +15.4] | +8.5 [+1.6, +15.2] | x0.85 |

Findings:
- **No arm keeps E034's pass@2.** The best short arm, D (low+header+summary), is 37% shorter (31.9k tokens)
  and 5.1 pp lower on pass@2. That is outside the noise band, although the CI includes 0. Its pass@1 is a
  tie (a0 +1.0, mean -1.9). Low effort mostly costs the second attempt: a1 is 57.9% for D vs 67.4% for a0.
- **Low effort alone hurts.** B and E are 10-13 pp lower on pass@2 than max and save only 17-25% of the tokens.
- **The summary is the useful lever at low effort.** It raises pass@1-mean by +8.5 to +10.6 pp and cuts
  tokens by 11-15% (C-B, D-E). At max (A, 28 easy tasks only) it is +5.4 pp on pass@2 at x0.94 tokens. That
  hints at a gain but was not measured on the full set.
- **The header is a tie on accuracy.** With the summary it saves 11% of the tokens (D-C pass@2 +1.8, pass@1
  mean -3.5). Alone it saves 13% (E-B -2.6 pp). This is unlike the tool harness, where E030's header cut 14% for free.
- Next candidate: the full run of A, max+summary (~$10 for 334 calls). Resuming A's file costs ~$7.

## Probe: API `reasoning_effort` levels (2026-10-20, ~$2)
The API now validates `reasoning_effort` (422 on unknown values) and accepts `none, minimal, low, medium, high, xhigh,
ultra, max`. On the same 15 eval pairs (no tools, plain ARC Prize prompt, thinking enabled, 1 call each; low = arm B
attempt 0), mean output tokens: low 48.2k, minimal 51.5k, medium 47.7k (none 53.0k: it still thinks); correct 7 / 8 / 8 / 9
of 15. Per-pair counts swing both ways with no trend, so `minimal` and `medium` behave like `low` here (DeepSeek's
low adds no prefix); the system-message header (arm D/E) is the only working "minimal" lever. Script and raw output:
session scratchpad `effort_probe.py` / `effort_probe_out.json` (not kept).

## All Flash-1017 public-eval runs, absolute numbers (`scripts/abs_table.py`)
Task-weighted; pass@1 = mean of attempts 0 and 1; tokens = output tokens per attempt (all turns for tool runs).

| Run | Tools | Calls | pass@2 | pass@1 | Out tokens mean / median / p90 / max | Correct-call median |
|---|---|---|---|---|---|---|
| E029 sub80s40 | tools | 334 | 82.4% | 74.3% | 31.7k / 27.5k / 64.1k / 80.0k | 21.9k |
| E030 sub80s40 + header | tools | 334 | 84.9% | 74.5% | 27.4k / 22.8k / 60.6k / 80.0k | 17.4k |
| E029 sub80s40 lean2code | tools | 334 | 81.8% | 73.8% | 32.9k / 29.5k / 67.0k / 80.0k | 22.9k |
| E030 cap16 | tools | 334 | 81.5% | 71.6% | 31.9k / 29.1k / 63.5k / 77.7k | 22.9k |
| E030 s20x80 | tools | 334 | 81.1% | 68.8% | 28.7k / 24.5k / 58.1k / 77.9k | 20.4k |
| E030 cap12 | tools | 334 | 79.4% | 69.4% | 31.0k / 24.6k / 62.5k / 72.6k | 19.9k |
| E030 s15x80 | tools | 334 | 78.3% | 65.7% | 27.9k / 24.3k / 56.8k / 80.0k | 19.4k |
| E030 s20x60 + header | tools | 334 | 80.3% | 69.9% | 23.3k / 20.6k / 44.0k / 60.0k | 16.1k |
| E029 s20x60 | tools | 334 | 77.8% | 65.8% | 26.7k / 24.5k / 51.4k / 60.0k | 18.1k |
| E034 max | none | 334 | 80.6% | 64.6% | 50.4k / 44.8k / 93.4k / 145.5k | 37.6k |
| E035 A max + stats (partial), 28 tasks | none | 93 | 91.1% (E034 85.7%) | 83.0% (E034 71.4%) | 45.4k / 40.9k / 85.7k / 124.8k | 34.3k |
| E035 C low + stats | none | 334 | 73.6% | 66.2% | 35.9k / 30.7k / 65.2k / 119.7k | 25.7k |
| E035 D low + header + stats | none | 334 | 75.4% | 62.6% | 31.9k / 28.0k / 57.8k / 114.4k | 22.4k |
| E035 E low + header | none | 334 | 67.9% | 54.2% | 37.6k / 31.2k / 72.2k / 124.8k | 24.7k |
| E035 B low (partial), 76 tasks | none | 226 | 74.6% (E034 84.2%) | 61.7% (E034 69.7%) | 40.4k / 33.7k / 74.4k / 135.8k | 26.8k |
