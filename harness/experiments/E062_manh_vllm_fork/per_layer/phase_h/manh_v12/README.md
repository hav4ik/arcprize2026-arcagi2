# Manh's submitted notebook (scriptVersionId 363638159): rerun-mode CV replica on pod 2

User request (2026-10-30 01:45 UTC, via the coordinator): run Manh's submitted notebook exactly as it is, in rerun mode,
on the 240-task x2t workload for the full ~12 h Kaggle clock. Track host RAM against Kaggle's limit, and VRAM, over the
whole run. Nothing is killed deliberately in this run.

## Setup
- **Notebook:** `kaggle/dsv41_tools/out/manh_v12_363638159/dsv41-tools-submit-b32m-external-server.ipynb`, sha256 c47be5a819ad, with its kernel-metadata.json.
  - His solver 11bfaaaa; his `dsv41_vllm.py --supervise` supervisor (status file `dsv41_logs/.../supervisor.json`, passed to the solver as ARC_EXTERNAL_SERVER_STATUS).
  - util 0.91, driver reserve 256 MiB, offload 10 GiB/rank, MANH_SEQS = ARC_WORKERS = 64, context 163840, ARC_MAX_SERVER_RESTARTS 3.
  - Docker on Kaggle: his BYOD image (`gcr.io/kaggle-private-byod/python@sha256:9a6bcfda…`). Here the kernel is our Kaggle-like Python 3.13.16 venv.
- **Runner:** `scripts/run_nb.py` (nbclient), started by `scripts/launch_nb.sh rerun m12_rerun <nb>` at 01:51:04 UTC.
  - KAGGLE_IS_COMPETITION_RERUN=1.
  - /kaggle/input/competitions/.../arc-agi_test_challenges.json is byte-identical to evalx2t_s2026 (240 tasks, 344 pairs).
- **Inputs** (`scripts/make_kaggle_tree.sh` with KT_* variables):
  - **Model:** the E128 Kaggle copy, chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1, 136 files. sha256 verified against the manifest embedded in the fix notebook.
  - **vllm-wheels v11:** the vllm wheel g8cf0c0664b (sha256 5039fd7680a4), plus 293 wheels identical to v10. The 4 nested source dirs are left out; no cell installs from them.
  - **arc2-2026-src v23**, the latest; pinned download. Its dsv41_vllm.py is 18,488 B and has --supervise; 167 files.
  - cu13-install v2 (5 epoch-colon debs from the Ubuntu archive, same names and sizes), dsv41-tools-kit v5, unsloth-main v15.
  - vLLM files: the wheel's originals. The kernel venv is reset to a fresh copy, cell 4 installs the wheels, and his notebook has no 2r.
- **Pod-only accommodation:** after the install cells (+43 s) the 20 GB uv cache in /tmp/dsv41/cache/uv is deleted for disk headroom (34 GB free).
- **Monitoring** (`scripts/v12_sampler.py`, read-only, every 60 s → `samp.jsonl`):
  - **Collected:** /proc/meminfo; RSS by group (TP workers, EngineCore, API server, his supervisor, solver, tool sandboxes, kernel); nvidia-smi per GPU; vLLM's per-rank "GPU memory:" lines; supervisor status; solver error counts.
  - **Kaggle-equivalent host RAM** = AnonPages + Shmem + 7 GiB. Kaggle's cgroup is 173.0 GiB with no swap.
  - **Alerts** (`samp.alerts`):

    | metric | WARN | ALERT | CRIT |
    |---|---|---|---|
    | Kaggle-eq RAM | ≥ 155 GiB | ≥ 165 GiB | ≥ 173 GiB |
    | VRAM free (any GPU) | < 1.0 GiB | < 0.5 GiB | |

    Also ALERT on: any CUDA OOM, allocator retry, engine death, supervisor restart, or stop=error attempt.
  - **Hourly:** `scripts/v12_report.py` (trend table in 30-min buckets, errors, accuracy, split table). `scripts/v12_wait.sh` copies the data to `manh_v12/data/` at each report.

## Final result (run ended 12:01:33 UTC, NOTEBOOK_DONE ok at +36,629 s = 10.17 h)
- **Kaggle score (task-weighted) 80.83; per pair 277/344 = 80.52%.**
  - Scored directly from the notebook's `submission.json` against evalx2t_s2026_solutions; `replica_report.py` gives the same.
  - **On Kaggle's clock: 80.83** (pessimistic 80.62, optimistic 80.83), cluster-bootstrap sd 3.1. Report: `replica_report.md`.
  - attempt_1 only: 75.62; attempt_2 adds +5.21.
- **Coverage:** all 240 jobs finished both attempts, the last at 10.17 h. The force deadline was not reached and no attempt was cut by a deadline. 338 of 344 pairs have an answer; 6 rows stay at the fallback.
- **Per-attempt accuracy** (finished attempts, per pair): attempt 0 72.1%, attempt 1 70.9%; train_pass True 83.4%.
- **Stability:** 0 engine deaths, 0 supervisor restarts (start 0 ran 36,576 s; the supervisor stopped cleanly in cell 4), 0 errored attempts, 0 APITimeoutError or HTTP 500, 0 allocator retries, 0 OOMs.
- **Memory over the whole run** (609 samples with the server up, 01:56-12:01):

  | measure | extreme | limit / reference |
  |---|---|---|
  | host RAM, Kaggle-eq (cgroup rss + shmem + 7) | 133.3-135.1 GiB (meminfo form max 148.0) | 173.0 GiB, so >= 38 GiB headroom |
  | VRAM cap - reserved (primary) | min 1.08 GiB (TP3), flat from 04:50 to the end | 0 |
  | VRAM cap - peak | min 1.58 GiB | 0.26 alert |
  | nvidia-smi free | min 1.32 GiB | board-level |
  | sandboxes RSS | max 3.4 GiB | |

- **Throughput:** 735-741 tok/s with 56-63 running in the first hour. Then KV-bound at 31-47 running, 510-630 tok/s (the solver's KV gate at 0.85, plus a server capacity queue at 1.5-2.5 h). Whole-run mean after the first 10 min: 561 tok/s at 37.2 running, including the idle tail after 9.5 h.
- **Data** (`data/`): executed notebook with every cell output (`nbruns/m12_rerun/nb.ipynb`, `nb_outputs.txt`), timeline, solver/server/supervisor logs, attempts, metrics, traces, submission.json, `samp.jsonl` / `samp.alerts`, `gpumem.log`, `ram.tsv`, `tps.tsv`, `status.json`, `workload.txt`.

## Results (as the run went)
- **Start:**
  - Install cells finished at +43 s.
  - Server up at +297 s: his supervisor, start 0; `supervisor.json` reads `{"state": "up", "up_s": 245.2, "max_restarts": 3}`.
  - KV pool **1,608,578 tokens** (util 0.91). Model loading 14.56 GiB per rank.
  - Solver from 01:56:02: "attempting 240 tasks, 344 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h".
- **Allocator cap** (startup, per rank): "Capped the PyTorch allocator at 21.15-21.20 GiB (19.15 GiB reserved), keeping 0.25 GiB free for other GPU allocations", so X - Y = 2.00-2.05 GiB.
  - Per the coordinator, pod VRAM is ~0.05 GiB optimistic vs Kaggle for the v11 stack. ALERT if cap - peak < 0.26 GiB.

### Report 1 (02:57 UTC; 1 h of solving; `report_0257.txt`)
- **Accuracy:** 54 first attempts, 46 correct (85.2%), all submitted.
- **Errors:** 0 errored attempts, deaths or restarts. 0 alerts.

| 30-min bucket end | Kaggle-eq max | AnonPages | Shmem | MemAvail min | RSS: TP / Eng / API / sup / solver / sandbox / kernel | nvidia-smi free min per GPU | vLLM alloc / peak / reserved / free (worst rank) |
|---|---|---|---|---|---|---|---|
| 02:20 | 145.4 | 131.4 | 7.1 | 347.9 | 127.8 / 1.4 / 1.4 / 0.0 / 0.5 / 3.3 / 0.1 | 1.68 1.79 1.81 1.77 | 18.85 / 19.18 / 19.62 / 1.77 |
| 02:50 | 145.5 | 131.6 | 6.9 | 347.9 | 127.8 / 1.4 / 1.4 / 0.0 / 0.7 / 3.4 / 0.1 | 1.67 1.63 1.65 1.62 | 18.85 / 19.27 / 19.78 / 1.62 |

| UTC | TP0: X-Y / cap-peak / cap-reserved | TP3: X-Y / cap-peak / cap-reserved | retries |
|---|---|---|---|
| 02:20 | 2.05 / 2.02 / 1.58 | 2.00 / 1.97 / 1.53 | 0 |
| 02:50 | 2.05 / 1.93 / 1.42 | 2.00 / 1.88 / 1.37 | 0 |

- **Load:**
  - gen 743-788 tok/s at 64 running for the first 30 min.
  - KV 0.82 at 27-32 min, 0.90 at 54-59 min. Running then fell to 39 with 5 waiting.
  - KV gate holds from 02:24; "server has 2-7 waiting requests" holds from 02:41.
  - Host KV restore starts at ~50 min (19-69% of prompt tokens); recompute 1.1-1.4%.

### Report 2 (03:57 UTC; 2 h of solving; `report_0356.txt`)
**Verdict: host RAM stable. VRAM headroom is shrinking slowly and steadily: cap - peak reaches 0.26 GiB in ~7 h at the current slope (~9 h of solving), before stop-new at 10.8 h.**

- **Measurement fix: /proc/meminfo on the pod is host-wide.** At 03:56:

  | source | anon / rss | shmem |
  |---|---|---|
  | /proc/meminfo | AnonPages 131.7 GiB | 6.9 GiB |
  | cgroup v1 memory.stat | total_rss 120.4 GiB | total_shmem 6.9 GiB |
  | summed process RssAnon / RssShmem | 120.4 GiB | 7.8 GiB |

  - So ~11.3 GiB of AnonPages is not ours. From 03:57 the sampler's `kaggle_eq_gib` is cgroup total_rss + total_shmem + 7 GiB, here 134.3 GiB; `kaggle_eq_meminfo_gib` keeps the old form (144.4).
  - The report-1 RAM figures use the meminfo form: subtract ~11 GiB for the cgroup view.
- **Growing (worst rank TP3, GiB):**

  | UTC | peak allocated | reserved | cap - peak | cap - reserved | retries |
  |---|---|---|---|---|---|
  | 02:20 | 19.18 | 19.62 | 1.97 | 1.53 | 0 |
  | 03:20 | 19.40 | 19.92 | 1.75 | 1.23 | 0 |
  | 03:56 | 19.47 | 19.98 | 1.68 | 1.17 | 0 |

  - Allocated is flat at 18.85; the ranks are within 0.05 of each other.
  - Slopes over 1.7 h, without the load ramp: cap - peak -0.19/h, cap - reserved -0.24/h.
  - Host: cgroup rss + shmem ~127 GiB, flat. TP-worker RSS 127.8 flat; sandboxes 2.9-3.4 GiB, not growing.
- **Close to a limit:**
  - cap - peak reaches 0.26 in ~7.3 h (~7 h with the pod's 0.05 GiB optimism).
  - cap - reserved reaches 0 in ~4.9 h. At the cap the allocator frees cached blocks, so a retry is the real signal; it alerts at once.
  - nvidia-smi free crosses 1.0 GiB in ~1.8 h. Board-level, so not a pressure signal.
  - RAM: 134 of 173 GiB (cgroup view), flat.
- **Events:** none. 0 OOMs, deaths, supervisor restarts (start 0), errored attempts or APITimeoutError.
  - KV-bound queueing: up to 19 waiting at the server; host KV restore up to 98% of prompt tokens at 70-80 min; gen/s 790 -> 404-690.
- **Accuracy:** 94 first attempts, 73 correct (77.7%); stops: 91 submitted, 3 output_budget.

### Status 04:26 UTC (2.5 h of solving): stable
- cap - peak: TP0 1.65, TP1 1.63, TP2 1.62, TP3 1.60 GiB. 0 retries, 0 OOMs.
- Kaggle-eq (cgroup): 133.8 GiB.
- First attempts: 120 done, 89 correct (74.2%). 0 errored attempts, 0 APITimeoutError or 500.
- Load: 28 running / 0 waiting; gen 496-535 tok/s.

### Throughput per 30 min of Kaggle clock (server 10 s lines; `scripts/v12_tput.py`)

| Kaggle clock (h) | mean gen tok/s | mean running | tok/s per stream | mean / max waiting | mean KV % | host prefix hit % | solver KV-gate holds (in flight at the hold) |
|---|---|---|---|---|---|---|---|
| 0.0-0.5 | 735 | 62.6 | 11.7 | 1.2 / 57 | 47.7 | 0 | 0 |
| 0.5-1.0 | 741 | 57.8 | 12.8 | 1.3 / 7 | 84.9 | 3.2 | 5 (63 → 53) |
| 1.0-1.5 | 547 | 32.2 | 17.0 | 3.4 / 6 | 88.4 | 86.0 | 6 (45 → 30) |
| 1.5-2.0 | 536 | 38.5 | 13.9 | 3.7 / 29 | 87.2 | 93.7 | 6 (28, 28, 62 → 53) |
| 2.0-2.5 | 491 | 28.1 | 17.5 | 1.3 / 3 | 86.9 | 95.9 | 6 (52 → 32) |
| 2.5-3.0 (partial) | 498 | 28.3 | 17.6 | 0.2 / 2 | 86.7 | 95.3 | 1 (29) |

Whole run after the first 10 min of serving: mean 646 gen tok/s, mean running 47.8.

**Why only ~28-32 requests run from hour 1 on: the solver's KV admission gate, which here is the KV capacity binding.**
- Once KV passes 0.85 (and later whenever the server has > 1 waiting), the solver admits no new attempts ("KV gate: … holding new attempts at N running"). Slots then empty one by one as long attempts finish: in flight 45 → 30 within 1.0-1.5 h.
- At ~47-53k tokens per stream, 30 streams x ~47k ≈ 1.41M ≈ 0.88 of the 1.61M pool. The concurrency the pool can hold is ~0.85 x 1.61M / (tokens per stream).
- **Not CPU restores:** the server's waiting queue averages 1-4.
- **Not tool calls:** the solver's in-flight count at the holds is only ~5 above the server's running count, so few attempts sit between turns.
- Host KV restore is heavy (86-96% of prompt-cache hits come from the CPU tier), but it serves re-prefills of evicted prefixes; it is not what limits concurrency.
- Per-stream speed rises (12 → 17.5 tok/s) as the batch shrinks, but total gen/s falls 740 → 490-550.

### Tighter watch from 04:28 UTC (coordinator)
- **Sampler:** 60 s, with a heartbeat file (`samp.heartbeat`) checked every round by `scripts/v12_wait.sh`. A heartbeat older than 180 s restarts the sampler and reports it.
- **Early warnings (`samp.alerts`):**
  - the first allocator retry or OOM;
  - cap - peak < 1.0 GiB, or a drop of > 0.15 GiB within 30 min;
  - Kaggle-eq > 150 GiB, or rising > 2 GiB/h;
  - any errored attempt, APITimeoutError or HTTP 500;
  - gen tok/s at 0 for 18 engine lines while requests are running;
  - server waiting > 30;
  - a supervisor status change, or a server-log ERROR / Traceback / 500;
  - the notebook runner exiting.
- **Reports:** a short one every 30 min, the full one hourly.

### Report 3 (04:57 UTC; 3.0 h of solving; `report_0456.txt`)
**Verdict: stable. VRAM headroom keeps shrinking linearly and slowly: worst cap - peak is 1.58 GiB, falling 0.16 GiB/h, so it reaches 0.26 in ~8 h (~11 h of solving, about at stop-new). Host RAM flat.**

| UTC | worst cap - peak | worst cap - reserved | retries / OOMs | Kaggle-eq (cgroup) | gen tok/s | running / waiting |
|---|---|---|---|---|---|---|
| 03:20 | 1.75 | 1.23 | 0 / 0 | ~135 (meminfo form 146.0) | 536 | 38 |
| 04:20 | 1.60 | 1.10 | 0 / 0 | 133.8 | 491 | 28 |
| 04:56 | 1.58 | 1.08 | 0 / 0 | 134.5 | 620 | 39-41 / 0-1 |

- **Growing:** peak allocated 19.40 -> 19.57 GiB in 1.6 h; reserved 19.92 -> 20.07. Allocated flat at 18.85.
- **Not growing:** cgroup rss + shmem; TP-worker RSS 127.8; sandboxes 2.4-3.3 GiB.
- **Events:** none (0 OOMs, deaths, restarts, retries, errored attempts or APITimeoutError).
- **Accuracy:** 138 first attempts, 97 correct (70.3%). Stops: 128 submitted, 6 output_budget, 3 max_turns, 1 context.
  - Accuracy falls as the long, hard attempts finish later.
- **Throughput:** 3.0-3.5 h: 620 tok/s at 41 running. Whole-run mean after 10 min: 621 tok/s at 44.7 running.

### Correction (05:59 UTC): throughput table and tps.tsv
- `v12_tput.py` and the tps backfill missed every Engine line carrying "KV fetch: …, …," before "GPU KV cache usage". That was a regex miss; both are fixed.
- Corrected values:

  | clock (h) | gen tok/s | running | server waiting |
  |---|---|---|---|
  | 0.5-1.0 | 729 | 55.9 | 2.0 |
  | 1.0-1.5 | 554 | 32.3 | 4.7 |
  | 1.5-2.0 | 565 | 40.2 | 6.9 (max 29) |
  | 2.0-2.5 | 540 | 32.1 | 10.6 (max 18) |

  Whole run after the first 10 min: 589 tok/s at 39.6 running.
- KV-fetch entries (waiting to start + in progress + completed waiting) average <= 0.9, so the server's waiting queue in 1.5-2.5 h was capacity (KV full), not CPU-restore fetches.
- The conclusion stands: KV capacity limits concurrency, through the solver's gate and, in 1.5-2.5 h, the server's capacity queue.
- `data/tps.tsv` now uses ISO utc. The only zero rows are 01:52-01:55 (server loading).

### Report 4 (05:56 UTC; 4 h of solving; `report_0556.txt`)
**Verdict: stable; VRAM growth has flattened.**
- Worst cap - peak 1.58 GiB, unchanged since 04:50: slope over 2 h -0.04 GiB/h, ~34 h to 0.26. Peak 19.57 and reserved 20.07, both flat. 0 retries, 0 OOMs.
- RAM: Kaggle-eq (cgroup) 133.9 GiB, flat. TP-worker RSS 127.8; sandboxes 1.7-3.2 GiB.
- Events: none.
- Accuracy: 182 first attempts, 127 correct (69.8%). Stops: 167 submitted, 11 output_budget, 3 max_turns, 1 context.
- Load: 4.0-4.5 h at 531 tok/s, 31 running, KV 0.88.

### Primary VRAM margin is cap - reserved (coordinator, 06:10 UTC, after pod 3's OOM)
- Pod 3 (B32-M) OOMed on a 32 MiB all-gather buffer with reserved 0.03 GiB below the cap (expandable segments), while cap - peak still read 0.40. **So cap - reserved, not cap - peak, is the danger signal.** My earlier "benign without retries" note was wrong.
- V12 at 06:10: cap - reserved 1.13 / 1.11 / 1.10 / 1.08 GiB (TP0-3); 2 h slope -0.016 GiB/h; ETA to 0 ~67-70 h; flat since 04:50.
- The sampler alerts at cap - reserved < 0.30 (EARLY) / 0.15 (ALERT) / 0.05 (CRIT), and on the first allocator retry. The reports lead with cap - reserved per rank, its slope and ETA.

### Report 5 (06:56 UTC; 5 h of solving; `report_0656.txt`)
**Verdict: stable; every memory margin flat for 2 h.**
- **VRAM:** cap - reserved TP0-3 = 1.13 / 1.11 / 1.10 / 1.08 GiB, slope 0.000/h over 2 h (last change ~04:50). cap - peak 1.58-1.63. 0 retries, 0 OOMs.
- **RAM:** Kaggle-eq (cgroup) 134.2 GiB, -0.14 GiB/h.
- **Events:** none.
- **Accuracy:** 225 first attempts, 162 correct (72.0%). Stops: 206 submitted, 15 output_budget, 3 max_turns, 1 context.
- **Throughput:**

  | window (h) | gen tok/s | running |
  |---|---|---|
  | 4.0-4.5 | 511 | 32.6 |
  | 4.5-5.0 | 619 | 43.8 |
  | 5.0-5.5 | 586 | 36.5 |

  Whole run after 10 min: 586 tok/s at 39.5 running.

### Report 6 (07:57 UTC; 6 h of solving; `report_0756.txt`)
**Verdict: stable; margins flat for 3 h.**
- **VRAM:** cap - reserved 1.13 / 1.11 / 1.10 / 1.08 GiB, slope 0.000/h, unchanged since 04:50. 0 retries, 0 OOMs.
- **RAM:** Kaggle-eq (cgroup) 134.2 GiB, flat.
- **Events:** none.
- **Progress:** all 240 tasks finished their first attempt by Kaggle 6.0 h: 171 correct (71.3%, scored on pair 0 per task). Stops: 220 submitted, 16 output_budget, 3 max_turns, 1 context. Second attempts are running (262 attempt records).
- **Load:** 5.5-6.0 h: 506-629 tok/s, 29-42 running, KV 0.84-0.90; host restore up to 94% of prompt tokens in some windows.

### Report 7 (08:57 UTC; 7 h of solving; `report_0856.txt`)
**Verdict: stable; memory margins flat for 4 h; no events.**
- **Accuracy trend** (`scripts/acc_report.py`, the dashboard's acc_series method: best of the finished attempts, all test pairs, all 240 tasks, unanswered = 0):
  - ARC score so far 75.6 at Kaggle 7.04 h; 233 tasks answered.
  - First attempts 166/222 = 74.8%; second attempts 24/53 = 45.3%.

  | by Kaggle h | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | score so far | 18.1 | 29.8 | 40.4 | 53.3 | 68.3 | 74.0 | 75.6 |
  | tasks answered | 50 | 87 | 126 | 164 | 206 | 224 | 233 |

  Earlier reports counted first-attempt records scored on pair 0 (e.g. 171/240 at 6 h); from here on, this metric.
- **VRAM:** cap - reserved 1.13 / 1.11 / 1.10 / 1.08 GiB, flat since 04:50. 0 retries, 0 OOMs.
- **Sampler bug fixed at 08:57:** the startup cap lines had scrolled out of the 3 MB log tail, so `alloc_cap` (and its alerts) was empty for 24 samples, the last ones ~08:33-08:56. Caps are now read from the log head once per log. In that window reserved stayed <= 20.07 GiB per rank (cap - reserved >= 1.08) with 0 retries, so nothing was missed.
- **RAM:** Kaggle-eq (cgroup) 134.7 GiB, +0.09 GiB/h.
- **Throughput:**

  | window (h) | gen tok/s | running |
  |---|---|---|
  | 6.5-7.0 | 555 | 37.6 |
  | 7.0-7.5 | 736 | 54.4 |

  The 7.0-7.5 h window is higher because second attempts start short, so the KV gate admits more. Whole run after 10 min: 578 tok/s at 38.1 running.

### Reports 8-9 (09:26, 09:57 UTC; 7.5-8 h of solving; `report_0956.txt`)
**Verdict: stable; margins flat for 5 h; no events.**
- **Accuracy:**

  | Kaggle h | score so far | tasks answered | first attempts | second attempts |
  |---|---|---|---|---|
  | 7.56 | 77.1 | 234 | 166/222 = 74.8% | 46/84 = 54.8% |
  | 8.05 | 79.2 | 235 | 166/222 = 74.8% | 76/116 = 65.5% |

- **VRAM:** cap - reserved 1.13 / 1.11 / 1.10 / 1.08 GiB, flat. 0 retries, 0 OOMs.
- **RAM:** Kaggle-eq (cgroup) 134.4 GiB, +0.22 GiB/h.
- **Throughput:**

  | window (h) | gen tok/s | running |
  |---|---|---|
  | 7.0-7.5 | 623 | 41.5 |
  | 7.5-8.0 | 627 | 47.1 |
  | 8.0-8.5 | 604 | 42.1 |

  Whole run after 10 min: 582 tok/s at 38.7 running.

### Reports 10-11 (10:27, 10:57 UTC; 8.6-9.1 h of solving; `report_1056.txt`)
**Verdict: stable; margins flat for 6 h; no events.**
- **Accuracy:**

  | Kaggle h | score so far | tasks answered | first attempts | second attempts |
  |---|---|---|---|---|
  | 8.58 | 80.2 | 235 | 74.8% | 96/144 = 66.7% |
  | 9.09 | 80.6 | 235 | 74.8% | 122/176 = 69.3% |

- **VRAM:** cap - reserved 1.13 / 1.11 / 1.10 / 1.08 GiB, flat since 04:50. 0 retries, 0 OOMs.
- **RAM:** Kaggle-eq (cgroup) 134.3 GiB, flat.
- **Throughput:**

  | window (h) | gen tok/s | running |
  |---|---|---|
  | 8.5-9.0 | 595 | 42.0 |
  | 9.0-9.5 | 559 | 38.2 |

  Whole run after 10 min: 582 tok/s at 38.9 running. Host restore ~0 since 8.5 h.
