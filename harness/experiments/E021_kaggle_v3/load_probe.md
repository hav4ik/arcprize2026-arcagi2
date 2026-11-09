# E021 load probe: v2 vs v4 config on 4xL4 (48 public-eval test pairs)

Kernels `chankhavu/arc2-load-probe-v2` and `-v4`, pushed 14:39 UTC 2026-10-13, both COMPLETE (v2 17:09, v4 ~17:20).
Outputs: `kaggle/runs/probe-v2`, `kaggle/runs/probe-v4`. Settings: `probe_v2_settings.json`, `probe_v4_settings.json`.
Analysis script: `scripts/probe_analyze.py <run dir>` (parses the kernel log, summary_local.json, attempts_local.jsonl and vllm_server.log).
Each run: the real scheduler, local mode on `ARC_LOCAL_SOURCE=eval`, 35 tasks / 48 test pairs, one attempt per pair
(first attempts only), force at server-ready + 120 min, hard stop +15 min. Times below are minutes after server ready.

## Side by side

| | v2 config | v4 config |
|--|--|--|
| GPUs | 4x NVIDIA L4 23 GB, TP4, vLLM 0.23.0 | same |
| server ready | 590 s after start (415 s of server boot) | 827 s after start |
| tier / answer mode / think cap | medium / submit_answer / none | xhigh / program (lean2progroles, task summary, helpers, rich feedback) / 8192 per turn |
| concurrency chosen | 35 (KV 1,402,470 tokens) | 29 (same KV) |
| scheduler run time | 132.1 min (force at 120 min after ready) | 120.0 min (all 48 attempts finished before force at 134 min after T0) |
| generated tokens | 1,090,988 | 1,544,685 |
| overall avg tok/s (first token to scheduler end) | **138** | **215** |
| tok/s in first 20 min (window mean) | 290 | 280 |
| tok/s min 20-60 | 189 | 256 |
| tok/s min 60-end | 77 (about 90 at min 60-110, 3 in the final drain) | 172 (declines as attempts finish; per running attempt about 10 tok/s) |
| engine log mean gen / prompt tok/s | 139 / 970 | 215 / 189 |
| per-attempt decode speed (median tokens/elapsed) | 4.1 tok/s | 9.6 tok/s |
| peak KV usage | **98.2%** (above 84% from min 55 to 120) | 84.7% |
| prefix-cache hit (2-min windows) | 92% at min 30, 46% at min 40, 8% at min 60, about 0-2% min 70-120 (overall client 9.8%) | 90-98% throughout after warm-up (overall 79%) |
| preemptions / vLLM warnings / tracebacks / OOM / restarts | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| max waiting requests | 25-34 only in the final forced-answer drain | 4 (progress), 26 (engine log, drain) |
| scheduler-assigned per-attempt budget | min 16,000, median 32,999, max 96,000 (first 35: 32,998-32,999; later starters: 34,419, 41,215, 85,076, 95,024, 96,000, 96,000, 96,000, 79,657, 43,677, then 16,000 x4 at min 69-102) | min 32,992, median 32,998, max 89,589 (first 29: 32,992-32,999; the 19 starts after min 18: 55,836-89,589) |
| budget actually run (capped by ARC_TOOLS_CAP 48,000) | 16,000 / 32,999 / 48,000 | 32,992 / 32,998 / 48,000 |
| attempts answered | 32/48 | 25/48 |
| answer sources | submit 17, forced 13, salvage 2, none 16 | program_pass 19, session_solve 5, program_best_fail 1, none 23 |
| ended by deadline force | 30 of 48 (plus 1 unclosed_think) | 3 of 48 (26 stopped at the output budget, 19 submitted) |
| tokens per attempt | mean 22.7k, median 28.1k, max 31.7k | mean 32.2k, median 33.0k, max 48.0k |
| attempt wall time | mean 91 min, median 109 min | mean 54 min, median 61 min |
| pass@1, pair-weighted | **5/48 = 10.4%** | **14/48 = 29.2%** |
| pass@1, task-weighted | **5.0/35 = 14.3%** | **11.5/35 = 32.9%** |
| correct by source | submit 5/17; forced 0/13; salvage 0/2 | program_pass 13/19; session_solve 1/5 |
| never attempted pairs | 0 | 0 |
| forced (deadline) pairs | 30 (none correct) | 3 |
| gated attempt 2 | off | 28 eligible, 0 started, 23 dropped for time ("no time left"), so no pass@2 in this probe |
| tool calls / errors | 496 / 24 | 382 / 66 |

## v2 time series (minutes after server ready, 10-minute rows; tok/s and hit are the mean of the last five 2-min windows, KV the max)

| min after ready | done | in flight | tokens | win tok/s | avg tok/s | rate est | KV % | cache hit % | prompt tok/2min | running | waiting | preempt |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 10 | 1 | 35 | 158065 | 264 | 263 | 220 | 41 | 70 | 434872 | 35 | 0 | 0 |
| 20 | 3 | 35 | 351695 | 322 | 293 | 293 | 56 | 89 | 288335 | 35 | 0 | 0 |
| 30 | 6 | 35 | 533429 | 303 | 296 | 316 | 68 | 92 | 343623 | 35 | 0 | 0 |
| 40 | 7 | 35 | 634227 | 168 | 264 | 261 | 76 | 46 | 265479 | 35 | 0 | 0 |
| 50 | 8 | 35 | 717906 | 140 | 239 | 199 | 80 | 28 | 164761 | 35 | 0 | 0 |
| 60 | 9 | 35 | 780163 | 104 | 217 | 132 | 84 | 8 | 727864 | 35 | 0 | 0 |
| 70 | 10 | 35 | 841938 | 103 | 200 | 117 | 88 | 2 | 980174 | 35 | 0 | 0 |
| 80 | 10 | 35 | 891487 | 83 | 186 | 97 | 92 | 0 | 400727 | 35 | 0 | 0 |
| 90 | 11 | 35 | 947046 | 93 | 175 | 90 | 94 | 2 | 223270 | 35 | 0 | 0 |
| 100 | 12 | 35 | 1000003 | 88 | 167 | 89 | 97 | 1 | 654386 | 35 | 0 | 0 |
| 110 | 15 | 33 | 1052679 | 88 | 159 | 91 | 95 | 12 | 1487840 | 33 | 0 | 0 |
| 120 | 18 | 30 | 1089193 | 61 | 151 | 79 | 93 | 2 | 1128536 | 0 | 0 | 0 |
| 130 | 41 | 7 | 1090821 | 3 | 140 | 50 | 14 | 4 | 4473945 | 3 | 4 | 0 |
| 132 | 47 | 1 | 1090988 | 3 | 138 | 50 | 14 | 3 | 4436207 | 1 | 0 | 0 |

## v4 time series

| min after ready | done | in flight | tokens | win tok/s | avg tok/s | rate est | KV % | cache hit % | prompt tok/2min | running | waiting | preempt |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 10 | 0 | 29 | 166452 | 277 | 277 | 220 | 37 | 4 | 307834 | 29 | 0 | 0 |
| 20 | 2 | 29 | 337017 | 284 | 281 | 281 | 51 | 90 | 382564 | 29 | 0 | 0 |
| 30 | 5 | 29 | 509334 | 287 | 283 | 283 | 60 | 94 | 286156 | 29 | 0 | 0 |
| 40 | 6 | 29 | 673160 | 273 | 280 | 277 | 71 | 95 | 291108 | 29 | 0 | 0 |
| 50 | 8 | 29 | 826621 | 256 | 276 | 272 | 80 | 92 | 416710 | 29 | 0 | 0 |
| 60 | 10 | 29 | 940871 | 191 | 261 | 239 | 84 | 57 | 270316 | 29 | 0 | 0 |
| 70 | 30 | 18 | 1054585 | 190 | 251 | 211 | 56 | 69 | 359592 | 18 | 0 | 0 |
| 80 | 30 | 18 | 1182979 | 214 | 246 | 195 | 44 | 96 | 211386 | 18 | 0 | 0 |
| 90 | 32 | 16 | 1300629 | 196 | 241 | 198 | 48 | 97 | 356723 | 16 | 0 | 0 |
| 100 | 36 | 12 | 1403282 | 171 | 234 | 191 | 48 | 97 | 284730 | 12 | 0 | 0 |
| 110 | 41 | 7 | 1486559 | 139 | 225 | 166 | 44 | 98 | 293783 | 7 | 0 | 0 |
| 118 | 43 | 5 | 1535665 | 106 | 217 | 140 | 28 | 98 | 223612 | 5 | 0 | 0 |

## Reading

- Both probes ran on 4xL4 as intended, with no preemptions, restarts or errors. The v4 config reproduces the public-eval level (about 29-33% pass@1).
  The v2 config reproduces the leaderboard level (10-14%), so the v2 collapse is reproducible on Kaggle hardware at the 12 h per-pair budget.
- v2 throughput and KV collapse: aggregate generation fell from about 300 tok/s (min 0-30) to about 90 tok/s from min 60. KV climbed to 84% by min 60 and 92-98% from min 80,
  all 35 first attempts having started together and growing in lockstep. As KV filled, the prefix cache was evicted
  (hit 92% to 0-2%), so every turn re-prefilled the whole 30-60k transcript (prompt throughput about 1,000-1,300 tok/s against about 90 generated).
  vLLM logged no preemptions: it evicted cached blocks, not running sequences, so the 0 preemption counter does not show the problem. Per-attempt decode
  fell to about 2.5 tok/s, so attempts reached only about 31k tokens in 2 h and the deadline forced 30 of 48. None of the 13 deadline-forced answers
  was correct (nor the 2 salvaged ones); only 5 of 17 self-submitted answers were.
- v4 at concurrency 29 stayed under 85% KV, kept 90-98% cache hits and about 10 tok/s per attempt, so attempts finished in about 54 min on average and
  only 3 hit the deadline. Whether the gain is from concurrency 29 vs 35 or from the config (think cap 8192 per turn, program answers, rich feedback, task summary)
  is not separated by this probe; the KV cliff sits between 84% and 92%.
- Budget shrinking is secondary. The scheduler's budget (rate estimate x time left / pending) was about 33k for the first wave in both, and the rate estimate fell
  to 50-90 tok/s in v2, giving four late attempts the 16,000 floor (all deadline-forced, no answer), while others were assigned 80-96k that TOOLS_CAP 48,000 clipped and the clock never allowed.
  So the budgets were not the binding constraint: the deadline was, via the collapsed throughput.
- v4 weakness for the real run: the gated retries never started (28 eligible, 23 logged "no time left" from 16:21 UTC, apparently because 0.85 x attempt speed x time left fell under B_MIN2 = 32,000; inferred from solver.py, not verified).

## Follow-up probes: v4b (concurrency 24, rate prior 250) and v4b-mtp2 (v4b + MTP speculative decoding, 2 draft tokens)

Same 48 pairs and scheduler. `probe_v4b_settings.json` differs from v4 only in `ARC_CONCURRENCY` 24 (v4 chose 29) and `ARC_RATE_PRIOR` 250 (v4 220);
`probe_v4b_mtp2_settings.json` differs from v4b only in `ARC_MTP` 2. Outputs: `kaggle/runs/probe-v4b`, `kaggle/runs/probe-v4b-mtp2`.
The mtp2 kernel's own T0 was 20:00 UTC (it sat queued after the 18:33 push, although it was listed as RUNNING); it then ran the normal about 2h10.

| | v4 (conc. 29) | v4b (conc. 24) | v4b-mtp2 (conc. 24, MTP 2) |
|--|--|--|--|
| server-reported KV capacity (tokens) | 1,402,470 | 1,402,470 | **1,171,281 (-16.5%)** |
| avg tok/s (first token to scheduler end) | 215 | **224** | 181 |
| tok/s min 20-60 / min 60-end (window mean) | 256 / 172 | 243 / 202 | 165 / 167 |
| engine-log mean gen / prompt tok/s | 215 / 189 | 224 / 162 | 181 / 665 |
| per-attempt decode (median) | 9.6 tok/s | 10.5 tok/s | 8.3 tok/s |
| peak KV | 84.7% | **65.2%** | **97.1%** (above 84% from minute 40 to about 95) |
| prefix-cache hit | 90-98% after warm-up (overall 79%) | 92-98% (overall 85%) | 2-10% in minutes 50-90, 33-77% elsewhere (overall 25%) |
| prompt tokens sent / generated | 15.3M / 1.54M | 16.8M / 1.57M | 26.9M / 1.30M |
| preemptions / errors / tracebacks | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MTP acceptance | n/a | n/a | mean acceptance length 2.39 (of max 3), draft acceptance 69% (position 1: 0.73, position 2: 0.58) |
| assigned budget, first wave | 32,992-32,999 (29) | 37,494-37,498 (24) | about 37.5k (24) |
| assigned budget, later attempts | 55.8k-89.6k (19; capped to 48k when run) | 31.2k-44.2k (24); one at 18.9k | min 16,000 (floor), max 51,163; median 37,495 over all |
| tokens per attempt (mean / median / max) | 32.2k / 33.0k / 48.0k | 32.7k / 36.5k / 44.2k | 27.2k / 27.3k / 48.0k |
| stops: output_budget / submitted / deadline | 26 / 19 / 3 | 26 / 22 / 0 | 34 / 13 / 1 |
| attempt wall time mean / max | 54 / 80 min | 52 / 73 min | 54 / 97 min |
| answered | 25 | 25 | 15 |
| pass@1 pair-weighted | 14/48 = 29.2% | **19/48 = 39.6%** | 7/48 = 14.6% |
| pass@1 task-weighted | 11.5/35 = 32.9% | **16.0/35 = 45.7%** | 7.0/35 = 20.0% |
| gated retries started | 0 (28 eligible, 23 dropped) | 0 (25 eligible, 25 dropped) | 0 (35 eligible, 34 dropped) |

v4b vs v4 (about 5 pairs of difference, within noise for 48 pairs): lower concurrency gave +4% tok/s, 20 points less peak KV, the same cache health, longer first-wave budgets
and 3 more correct pairs. MTP 2 is a regression here: it cut KV capacity by 16.5%, and with the same concurrency the KV went to 97% and the prefix cache collapsed
(the same cliff as v2; 26.9M prompt tokens were re-prefilled), so throughput fell below v4b even with 69% draft acceptance, and 34 of 48 attempts ran out of tokens.

### v4b time series
| min after ready | done | in flight | tokens | win tok/s | avg tok/s | rate est | KV % | cache hit % | prompt tok/2min | running | waiting | preempt |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 10 | 0 | 24 | 160803 | 268 | 268 | 250 | 31 | 7 | 204208 | 24 | 0 | 0 |
| 20 | 1 | 24 | 319393 | 264 | 266 | 266 | 44 | 92 | 318014 | 24 | 0 | 0 |
| 30 | 3 | 24 | 469269 | 250 | 261 | 261 | 52 | 88 | 318897 | 24 | 0 | 0 |
| 40 | 7 | 24 | 612069 | 238 | 255 | 247 | 59 | 92 | 265105 | 24 | 0 | 0 |
| 50 | 10 | 24 | 758362 | 244 | 253 | 242 | 61 | 95 | 314344 | 24 | 0 | 0 |
| 60 | 14 | 24 | 890031 | 220 | 247 | 232 | 63 | 92 | 326068 | 24 | 0 | 0 |
| 70 | 24 | 24 | 1039492 | 249 | 247 | 238 | 47 | 83 | 270926 | 24 | 0 | 0 |
| 80 | 24 | 24 | 1187179 | 246 | 247 | 238 | 59 | 96 | 396864 | 24 | 0 | 0 |
| 90 | 28 | 20 | 1322806 | 226 | 245 | 240 | 65 | 94 | 319618 | 20 | 0 | 0 |
| 100 | 35 | 13 | 1439143 | 194 | 240 | 219 | 56 | 97 | 259213 | 13 | 0 | 0 |
| 110 | 39 | 9 | 1529973 | 151 | 232 | 187 | 47 | 98 | 260390 | 9 | 0 | 0 |
| 116 | 46 | 2 | 1570430 | 121 | 226 | 164 | 36 | 98 | 222333 | 2 | 0 | 0 |

### v4b-mtp2 time series
| min after ready | done | in flight | tokens | win tok/s | avg tok/s | rate est | KV % | cache hit % | prompt tok/2min | running | waiting | preempt |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 10 | 0 | 24 | 201880 | 336 | 336 | 250 | 52 | 33 | 253263 | 22 | 0 | 0 |
| 20 | 1 | 24 | 344612 | 238 | 287 | 287 | 68 | 61 | 518701 | 24 | 0 | 0 |
| 30 | 4 | 24 | 453448 | 181 | 252 | 252 | 74 | 48 | 661867 | 24 | 0 | 0 |
| 40 | 6 | 24 | 607910 | 258 | 253 | 238 | 84 | 73 | 523682 | 24 | 0 | 0 |
| 50 | 6 | 24 | 663921 | 93 | 221 | 175 | 89 | 2 | 746975 | 24 | 0 | 0 |
| 60 | 8 | 24 | 717717 | 90 | 199 | 138 | 88 | 9 | 555667 | 24 | 0 | 0 |
| 70 | 9 | 24 | 794265 | 128 | 189 | 105 | 93 | 3 | 502654 | 24 | 0 | 0 |
| 80 | 10 | 24 | 856745 | 104 | 178 | 108 | 96 | 0 | 603436 | 24 | 0 | 0 |
| 90 | 22 | 24 | 945521 | 148 | 175 | 126 | 91 | 10 | 370672 | 24 | 0 | 0 |
| 100 | 26 | 22 | 1115278 | 283 | 186 | 182 | 71 | 77 | 289012 | 21 | 0 | 0 |
| 110 | 40 | 8 | 1261167 | 243 | 191 | 233 | 69 | 93 | 271429 | 8 | 0 | 0 |
| 118 | 47 | 1 | 1301366 | 104 | 184 | 215 | 37 | 98 | 132169 | 1 | 0 | 0 |

### Why gated retries are dropped (verified from `kaggle/src/solver.py` and the logs)

`next_job` serves `pending1` first and considers attempt 2 only when `pending1` is empty. `gated_budget` then returns 0 with reason "time" when
`ATTEMPT2_SAFETY (0.85) x attempt_speed x (force_at - 120 s - now) < B_MIN2 (32,000)`, and the whole `pending2` queue is cleared ("no time left ... dropping them"), since time only shrinks.
In these probes 48 pairs on 24-29 slots are only two waves, so `pending1` empties after about 61 min (ready-relative):
- v4: 16:21:34 UTC, speed 9.0 tok/s, 58 min left: 0.85 x 9.0 x 3,480 s = 26.6k < 32k.
- v4b: first drop 20:07:13 (the queue emptied about 19:43), speed 10.4 tok/s, 31.5 min left: 0.85 x 10.4 x 1,893 s = 16.7k < 32k.
  Even at the moment the queue emptied the figure was about 0.85 x 10.4 x 3,400 s = 30k, just under 32k.
So retries need about 60 min of runway at roughly 10 tok/s, and this 2-wave probe leaves 57 min. This is a probe artifact: in the 12 h rerun (259 pairs) the first-attempt queue empties at about 7.7 h of 11 h, leaving about 3 h.
No retry ever started in any probe, so gated retries are untested on Kaggle.
