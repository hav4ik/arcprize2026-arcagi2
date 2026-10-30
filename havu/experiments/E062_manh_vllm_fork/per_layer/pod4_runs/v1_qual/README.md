# Pod 4: B32-M fix V1 qualification replica with fault injection (2026-10-30)

**Verdict: stopped by the user at Kaggle 6.51 h and dropped ("V1 is very bad, Manh V12 is gold standard"). Not qualified,
because the run did not finish its 12 h.** Correction (coordinator, 2026-10-30 16:30 UTC): V1 already trailed V12 before
the first fault. It started serving ~4 min later (first throughput point 0.15 h vs 0.09 h), ran 12% slower in its first
half-hour, and had answered 116 tasks at 3 h vs V12's 126 (score so far 35.7 vs 40.4; B32's 10-27 replica: 92, 31.9).
Per-task accuracy ties (173 shared tasks).

**Update (SLOWDOWN.md, 2026-10-30 evening): the pre-fault gap is pod 4's host, not V1's notebook.** The same notebook
(bec71fb1) on pod 2 the night before matched V12 (server cell 241 vs 246 s; decode 1.003x at matched load, clean for
0.58 h). On pod 4: startup compile 237 vs ~150 s, TP3 (other NUMA node) loaded 55 s late, decode 0.937x per step, first
prefill burst 16% slower, and **0.70x after the 3 h restart**. Throughput explains ~6 of the 10-task gap at 3 h; the
rest is completion order. Our layer (supervisor, solver diff, util 0.90, --language-model-only) showed no measurable
effect. The post-kill explanation below (longer restarted contexts) is superseded: pod 4 itself decoded slower.

What the 6.5 h showed:
- **Stable.** The VRAM margin shrinks slowly, and host RAM is flat. 0 OOMs, 0 allocator retries, no unplanned death.
- **Both faults recovered** with 0 attempts finalized as error and never two engines.
- **Each restart is expensive.**
  - Kill-tp: about 40 min of serving, for a 176 s outage.
  - Hang: about 78 min of serving, plus 33 tries burned.
  - The cause is that every in-flight attempt restarts from scratch.
- **Score:** 62.15 Kaggle-style at the stop. Manh's V12 replica had 68.3 at 5 h and 74.0 at 6 h on the same workload, and the gap is all after the kill.
- **Fixes:** R3, R4 and R9 in `kaggle/dsv41_tools/next_candidate_fixes.md` address what the faults exposed.

Coordinator plan: setup, then a power test, then a 12 h rerun-mode replica with kill-tp at 3.0 h and sigstop-engine at
6.0 h, with hourly reports. Scripts: `../../pod4_scripts/`. Dashboard files: `data/`. Raw run copy: `raw/`. Larger files
are on HF (see [Files](#files)).

## Pod
- **Pod:** Runpod pod 4 = 9u4jdy2rczh9fu, 4x L4 at 72 W (min 40 / max 72 W), driver 595.91.07, CUDA 13.0.
- **Host:** 128 CPUs, 692 GB host RAM, cgroup v1 limit 321 GiB, 200 GB disk, /dev/shm 161 GB.
- **Topology:** GPU0-2 are NODE to each other. GPU3 is on the other NUMA node, SYS to every other GPU.
  - This differs from pods 2 and 3, so throughput is not strictly comparable across pods.
- **Access:** SSH address read from the Runpod GraphQL API (read-only). The pod was never stopped or modified through the API.

## Candidate and inputs (all checked)
- **Notebook:** `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/dsv41-tools-submit-b32m-fix.ipynb`, sha256
  bec71fb1e3f139d0 (Kaggle chankhavu/dsv41-tools-submit-b32m-fix V1).
  - Server: Manh's v6 cell 2m, TP4, no DCP, util 0.90, 64 seqs, offload 10 GiB/rank.
  - Our cell 2s supervisor: at most 2 restarts, stall limit 1500 s.
  - Solver 1ea2b25d, harness 882c817f.
- **Model:** E128, chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1, version-pinned per file from Kaggle.
  - Two streams, 136 files, 131.09 GB in 15 min (~150-250 MB/s).
  - `VERIFY OK: 136/136` against the sha256 manifest embedded in the notebook's cell 1b.
- **vllm-wheels v11** (pinned by kernel-metadata): 294 wheels per file, size-checked.
  - vllm-0.32.1rc1.dev0+g8cf0c0664b, sha256 5039fd7680a4.
  - The 4 nested source dirs are left out, as on pods 2 and 3; no cell installs from them.
- **arc2-2026-src v23:** all 128 top-level files are byte-identical to the copies in the Kaggle V1 commit run's working
  dir, including dsv41_vllm.py 638c35cd (v22 lacks `--supervise`).
- **dsv41-tools-kit v5:** dsv41_kit.tgz f24e624b.
- **cu13-install v2:** 107/107 files at the listed size. 102 came from Kaggle; the 5 epoch-colon debs came from the Ubuntu
  archive.
  - pod 2's dpkg fix was applied (`fix_cu13_deps.sh`); dpkg pass 2 rc=0, nvcc 13.0.88.
- **unsloth-main:** not mapped. Nothing reads it; its installs fail harmlessly on Kaggle too.
- **Workload:** evalx2t_s2026 (240 tasks, 344 pairs) as the test file in rerun mode. See `data/workload.txt`.
- **Kernel venv:** /root/kenv (Python 3.13.16, pod 2's 48-package fresh freeze), reset from a pristine copy before each
  run. Manh's install cells install the v11 stack into it, as on Kaggle.

## Runner and monitoring
- **Runner:** `kaggle/reliability/run_notebook.py` (nbclient), started by `launch_run.sh`.
  - Clean `env -i`-like environment; `/kaggle/input` mapped by symlinks with `--prune-input`; a fresh working dir.
  - `--mode rerun` sets KAGGLE_IS_COMPETITION_RERUN=1.
- **Kit monitors on the pod:**
  - `sampler.py` (60 s, 5 s when the server is down or after a fault);
  - `watcher.py --follow`;
  - a /metrics scraper every 60 s (`scrape_loop.sh`).
- **Local monitoring:**
  - `watch_loop.sh`: the watcher `--once` every 60 s, plus status checks on cap - reserved, deaths, errors and the supervisor;
  - `export_dash.sh` every 5 min into `data/` (gpumem.log, ram.tsv, tps.tsv, status.json, attempts.jsonl, solver.log);
  - `hourly.py`, `acc_report.py` (the team's acc_series.py) and `cmp_v12.py` for the reports.
- **Deviations from the exact notebook (pod-only):**
  1. **uv cache deleted:** `rm -rf /tmp/dsv41/cache/uv` as cell 12 starts, as pod 2 did. That is 20 GB of disk; nothing after cell 11 uses uv.
  2. **No GPU hold.**
     - Cell 2s waits for an empty `nvidia-smi --query-compute-apps` list before it relaunches, so a hold process would block every restart. This is R5 in the fix list.
     - It is also not needed: at startup cap - reserved is 2.26 / 2.24 / 2.23 / 2.21 GiB per rank, against the Kaggle V1 commit run's 2.22 / 2.21 / 2.19 / 2.17. The pod has +0.04 GiB; the calibration would call for a 40 MiB hold.
  3. VLLM_GPU_MEMORY_LOG_INTERVAL is already 60 in the notebook's MANH_SERVER_ENV, so no env change was needed.
- **Power test: skipped, not permitted.** `nvidia-smi -i 0 -pl 72`, setting the current value, fails with "Insufficient
  Permissions" (rc 4), so the 50 W question stays open on Runpod.

## Commit-mode smoke (09:10-09:20 UTC, `../commit1/`)
- **Server:** MANH SERVER UP after 390 s; KV 1,617,253 tokens (Kaggle V1: 1,614,903); model loading 14.33 GiB/rank.
- **Checks:** E046 CHECK OK (solver 1ea2b25d); SANDBOX CHECK OK; contract passes (/tokenize 377, token ids 64); INPUT INTEGRITY OK (136 files).
- **Supervisor:** watching, then stopped by cell 3s.
- **Light smoke:** 4/4 attempts answered, all correct on the public training solutions (the same test inputs).
- **Verdict cell:** raised only on the known cgroup-v1 RAM check (`max() iterable argument is empty`), as on pod 2. Kaggle is v2.

## Replica timeline (rerun mode; notebook start 09:21:10 UTC = Kaggle T0)
| UTC | Kaggle h | event |
|---|---|---|
| 09:29:15 | 0.13 | server up after 405 s; solver starts on 240 tasks / 344 pairs; deadlines stop-new 10.83 h |
| 12:21:31.8 | 3.00 | **fault 1: SIGKILL `VLLM::Worker_TP3`** (pid 19859, GPU 3) with 34 running |
| 12:24:28 | 3.05 | server up again (start 1): 176 s outage |
| 15:21:32 | 6.00 | **fault 2: SIGSTOP `VLLM::EngineCore`** (pid 35013) with 33 running, left frozen |
| 15:36:28-33 | 6.25 | 33 APITimeoutErrors (900 s client read timeout): 33 tries spent |
| 15:46:39 | 6.42 | supervisor: "no generated tokens for 1504s with requests running; restart 2/2" |
| 15:50:15 | 6.48 | server up again (start 2): 28.7 min outage |
| 15:51:39 | 6.51 | **stopped by the user** (via the coordinator): runner, kernel (supervisor thread), solver + 64 sandboxes and server killed by process group; GPUs at 0 MiB; `raw/runs/v1_qual/stop.log` |

## Stability (`reli_report.md`; the kit's verdict is N/A because the last server lived only 1 min)
- **Host RAM:** Kaggle-eq (cgroup v1 anon + shmem + 7 GiB) 131.6-133.7 GiB, flat; peak 133.75 against 173. One engine
  throughout, also through both restarts. Sandboxes peaked at 64, 3.4 GiB.
- **VRAM:** 0 allocator retries and 0 OOMs on every server. Cap - reserved per rank (TP0 / TP1 / TP2 / TP3, GiB):

  | server | UTC | cap - reserved | cap - peak | peak | reserved |
  |---|---|---|---|---|---|
  | 0 | 09:29 (up) | 2.26 / 2.24 / 2.23 / 2.21 | 2.53 / 2.51 / 2.50 / 2.48 | 18.67 | 18.94 |
  | 0 | 10:29 | 1.60 / 1.58 / 1.57 / 1.55 | 2.12 / 2.10 / 2.09 / 2.07 | 19.08 | 19.60 |
  | 0 | 11:29 | 1.40 / 1.38 / 1.37 / 1.35 | 1.92 / 1.90 / 1.89 / 1.87 | 19.28 | 19.80 |
  | 0 | 12:21 (killed) | 1.36 / 1.34 / 1.33 / 1.31 | 1.89 / 1.87 / 1.86 / 1.84 | 19.31 | 19.84 |
  | 1 | 12:29 | 1.81 / 1.79 / 1.78 / 1.76 | 2.26 / 2.24 / 2.23 / 2.21 | 18.94 | 19.39 |
  | 1 | 13:19 | 1.64 / 1.62 / 1.61 / 1.59 | 2.17 / 2.15 / 2.14 / 2.12 | 19.03 | 19.56 |
  | 1 | 14:19 | 1.50 / 1.48 / 1.47 / 1.45 | 2.02 / 2.00 / 1.99 / 1.97 | 19.18 | 19.70 |
  | 1 | 15:19 (frozen) | 1.34 / 1.32 / 1.31 / 1.29 | 1.86 / 1.84 / 1.83 / 1.81 | 19.34 | 19.86 |
  | 2 | 15:51 | 1.81 / 1.79 / 1.78 / 1.76 | 2.26 / 2.24 / 2.23 / 2.21 | 18.94 | 19.39 |

  - Allocated stays at ~18.6 GiB. The creep is reserved growing with the peak: cache, not a leak.
  - Server 0 flattened by its 3rd hour: -0.66 in hour 1 including warm-up, then -0.20, then -0.04 GiB/h.
  - **Server 1 did not flatten in 3 h** (-0.17 / -0.14 / -0.16 GiB/h). At that rate one server would reach cap - reserved ~0 after ~11 h, about the run's length.
  - Manh's V12 at util 0.91 flattened at 1.08 GiB after ~3 h. This is a watch item for a full run, not a failure here (R1: keep ≥ ~0.75 GiB).
- **Throughput** per half-hour of Kaggle clock (gen tok/s, running streams, waiting, KV %):

  | Kaggle h | 0-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-2.5 | 2.5-3 | 3-3.5 | 3.5-4 | 4-4.5 | 4.5-5 | 5-5.5 | 5.5-6 | 6-6.5 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | gen tok/s | 673 | 702 | 576 | 490 | 524 | 527 | 407 | 483 | 414 | 376 | 340 | 406 | 1 |
  | running | 63 | 60 | 38 | 32 | 37 | 35 | 55 | 56 | 37 | 27 | 34 | 35 | 27 |
  | waiting | 0.5 | 0.6 | 4.9 | 1.5 | 1.3 | 2.0 | 2.8 | 3.6 | 6.1 | 6.5 | 0.8 | 2.5 | 2.1 |
  | KV % | 43 | 81 | 89 | 87 | 87 | 88 | 53 | 88 | 89 | 88 | 87 | 87 | 83 |

  KV-bound from ~1 h (the 0.85 gate). The 6.0-6.5 window is the frozen engine.

## Fault 1: kill-tp 3 at 3.0 h: PASS (`fault_inject.py verify`)
- **Target:** exactly one process titled `VLLM::Worker_TP3` in the API server's tree. Dry run first; pidfd signal.
- **Timeline after the kill:**
  - +11 s: API server exits; solver logs `server died`.
  - +14 s: GPUs free; supervisor state "restarting".
  - +21 s: all old processes gone.
  - +26 s: first new process.
  - +176 s: supervisor "up" (start 1, `up_s` 150), solver `server up after 165s`, the same KV pool.
- **Checks:** finalized_errors 0, server_died, requeued 83, restart_time 176 s (limit 900), tries_spent 0, one_engine (Kaggle-eq 132.4 vs 133.6 before), ram_limit: all PASS.
  - 31 jobs were re-queued; 27 finished normally later and 4 were still running at the stop.
- **Cost:**
  - The 31 re-queued attempts restart from scratch, which threw away **1.16M generated tokens** (28.1 attempt-hours, about 38 min of the server's whole output).
  - Throughput then stayed ~30% below V12 for 2.5 h, because the restarted long attempts filled the KV again (22-27 streams at ~54k live context each, against V12's 35-44 at ~32k).
  - Score so far: 35.69 just before and 45.07 1 h after.
- **Finding R3:** in the ~5 s between the API server's exit and the supervisor's "restarting" (it polls every 5 s), the solver resubmitted re-queued jobs straight into "Connection refused", with no backoff. 8 jobs used 5 and 6 jobs used 4 of their 12 ARC_CONN_TRIES in one outage.

## Fault 2: sigstop-engine at 6.0 h: recovered, WARN (tries spent)
- **Timeline:**
  - SIGSTOP at 15:21:32; output went to 0 tok/s and /metrics kept answering (running 31).
  - +900 s: **33 APITimeoutErrors**; every in-flight attempt spent a try (`retry`, now 2/2 used) and restarted from scratch into the frozen server.
  - +1504 s: the supervisor's stall detection fired; SIGTERM; the API server exited within ~5 s; no old processes or GPU apps left.
  - +1722 s: `server up after 210s (external, start 2)`; 16 connection errors were re-queued for free at the restart.
- **Checks:** finalized_errors 0, server_died, requeued 16, restart_time 1722 s (limit 3000), one_engine, ram_limit: PASS.
  - **tries_spent: WARN (33).** Those 33 jobs had not finished by the stop and were one timeout away from being finalized as errors.
- **Cost:** a 28.7 min outage plus **1.19M generated tokens** thrown away (45.5 attempt-hours), about 78 min of serving in all.
  The restart budget (2) was used up, so a third death would have ended serving.
- **Finding R4:** solver 1ea2b25d counts APITimeoutError as a try, and the stall timer (1500 s) is longer than the client read
  timeout (900 s), so a hang always burns a try per in-flight attempt before the restart.

The optional gpu-oom-burst at 8.5 h was not run: the run was stopped at 6.51 h, and the restart budget was already spent.

## Score and accuracy
- **At the stop (6.51 h):** submission.json scores **62.15** Kaggle-style (task-weighted; `replica_report.md`, cluster-bootstrap sd 3.8).
  - 211/344 pairs right; 61 pairs still at the fallback.
  - The replica report's Kaggle-clock correction does not apply: the run stopped before the force deadline.
- **First attempts:** 200 done, 139 fully correct; per-pair accuracy of finished attempts 74.6% (train_pass True: 80.2%).
- **Score so far** (acc_series.py: best of the finished attempts, all pairs, unanswered = 0), V1 vs Manh's V12 replica:

  | Kaggle h | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | V1 | 14.58 | 27.15 | 35.69 | 45.07 | 51.53 | 62.36 |
  | V12 | 18.12 | 29.79 | 40.42 | 53.33 | 68.33 | 73.96 |
  | V1 tasks answered | 43 | 85 | 116 | 141 | 162 | 193 |

## Why V1 trails V12 (`../../pod4_scripts/cmp_v12.py`, data to 5.2 h)
- **Server config:** the launch commands are identical (TP4, no DCP, max-num-seqs 64, max-num-batched-tokens 2048,
  cudagraph FULL capture 64, fp8 KV, block 128, offload 10 GiB/rank, retention 128, the same MANH_SERVER_ENV). The only differences:
  - util 0.90 vs 0.91 (KV 1.617M vs 1.609M);
  - V12 keeps the vision tower (`--limit-mm-per-prompt image:16`) where V1 runs `--language-model-only`; prompts are text in both;
  - V1's VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS 900.
- **Harness:** identical (882c817f), with the same ARC_* settings (64 workers, multi-test 0.5, context 163840, KV gate
  0.85 / min 16, ARC_ADMIT_MAX_WAITING 1, auto-repair, deadlines). The solvers differ only in restart handling.
- **Per-attempt work** on the 173 tasks both runs finished:

  | | generated p50 / p90 | turns p50 / p90 | context at last turn p50 / p90 |
  |---|---|---|---|
  | V1 | 29.0k / 80.0k | 14 / 34 | 45.5k / 95.9k |
  | V12 | 28.1k / 81.1k | 13 / 33 | 47.7k / 109.9k |

- **First-attempt accuracy** on the same 173 tasks: 68.8% vs 69.9% (11 vs 13 discordant). A tie.
- **Throughput:** 0-3 h averaged 582 vs 592 tok/s. After the kill, V1 averaged ~400 tok/s against ~560, with the same ~14 tok/s per stream.
- **The gap** was 3-5 points before the kill (completion order) and 8.2 / 16.8 at 4 / 5 h.

**Conclusion:** most of the gap after 3 h is the restart cost; the 3-5 points before it are not just noise. V1 started
slower and finished ~8% fewer tasks by 3 h, plausibly from our added supervisor/solver layer (unmeasured).

## Fixes this run supports (`kaggle/dsv41_tools/next_candidate_fixes.md`)
- **R3:** a refused connection must not spend a connection try. Back off, or wait for the supervisor's "up".
- **R4:** APITimeoutError (and HTTP 500) must be free re-queues (solver a9983f4c / 64ef0a0f), with the stall-restart timer below the 900 s read timeout.
- **R9:** resume a re-queued attempt from its kept token history (one re-prefill) instead of regenerating it from scratch. A restart then costs minutes instead of ~40-80 min of serving.
- **R5:** before a relaunch, the supervisor should wait only for the old server's own processes, not for an empty GPU compute-app list.
- **R1:** watch the margin. Server 1's cap - reserved fell ~0.15 GiB/h for 3 h without flattening.

## Files
- **`data/`:** the dashboard files, same formats as pod 3: gpumem.log, ram.tsv, tps.tsv, status.json, attempts.jsonl, solver.log, workload.txt.
- **`raw/`:** the run copy, files under 1 MB.
  - `runs/v1_qual/`: timeline, nb.log (cell outputs, including the supervisor lines), nb.ipynb, manifest, fault.jsonl, alerts, metrics_scrape.txt, stop.log.
  - `kwork_v1_qual/`: dsv41_logs (attempts, solver.log, metrics, status, supervisor json, launch.json per start, r2's server.log) and submission.json.gz.
  - `p4/logs/`: download, verify and setup logs.
- **Reports:** `replica_report.md` (E055 toolkit, Kaggle-style score; run on a flat copy of dsv41_logs + submission.json), `reli_report.md` / `reli_report.json` (kit report with the fault checks).
- **On HF**, private dataset `livobereshka/arc-agi2-vllm-serving-lab`, `replicas/v1_qual/` (commit 9dfd4ecbf38f; checked private; secret-scanned, 0 hits): see `RAW_README.md`.
- **Commit smoke:** `../commit1/`.
