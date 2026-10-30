# B32-M V3 replica on pod 3 (rerun mode, x2t workload, full Kaggle clock)

**Goal (user, 2026-10-30):** re-run the B32-M submission exactly as submitted, for its full ~12 h Kaggle clock, in rerun mode on the 240-task evalx2t_s2026 workload, with Kaggle-like VRAM. It scored **46.5** on Kaggle against ~77.9 on an earlier pod replica.
The memory auditor (scratchpad `audit_b32m_mem/notes.txt`) predicts a GPU OOM, not a host OOM, at ~3.1-4.4 h pod time once VRAM matches Kaggle's.
No deliberate kills: if it OOMs, the solver's own restart logic plays out as on Kaggle.

Scripts: `pod3_scripts/b32mv3/` (on the pod: `/root/e066/b32mv3/`).

## Notebook and inputs (as at the submission, 2026-10-28 23:37 UTC)
- **Notebook:** `kaggle/dsv41_tools/out/b32m_v3_363310806/dsv41-tools-submit-b32m.ipynb`, sha256 771da13fd94d3746 (scriptVersionId 363310806), plus its kernel-metadata.json.
  - Contents: B32's harness and solver (the pinned solver in cell 0b is 77fe80a3; it launches its own server on port 8100 and restarts it up to 3 times), DCP4 + interleave 64, util 0.945 (the settings cell), 32 GiB CPU KV offload (OffloadingConnector), 80 workers, max-num-seqs 96, and cell 2r's Engram row-chunk patch (base 5e1bccd07c, ead1e087 -> b2a04fc2).
- **Dataset versions:** the notebook does not pin them, so they are the versions current at 23:37 UTC (Kaggle API version dates).

  | input | version | notes |
  |---|---|---|
  | arc2-2026-src | **v22** | created 10-28 05:41; v23 is from 10-29 21:22 |
  | vllm-wheels | **v9** | created 10-28 20:08. Its full listing (1,669 files) equals v8's by name and size. |
  | cu13-install | v2 | |
  | dsv41-tools-kit | v5 | dsv41_kit.tgz sha256 f24e624b |
  | unsloth-main | not mapped | training-only installs; those `!uv pip install` lines fail without raising |

  - vllm-wheels detail: the vllm wheel is vllm-0.32.1rc1.dev0+g5e1bccd07c (291,052,491 B, sha256 85e05736…). The other 293 top-level wheels equal v10's, so `/root/b32m/dl9` is v10's hardlinks plus that wheel and .gitattributes, downloaded pinned to versions/9 (`prep_v9.sh`). The 4 nested source dirs (1,374 files) are left out, as on pods 2 and 3 before.
  - cu13-install detail: `/root/kin/cu13` (107 files, 4.1 GiB) has 102 files per file from Kaggle; the 5 epoch-colon names, which the API answers with 404, come from the Ubuntu 24.04 archive via `apt-get download` (`c1text/fetch_cu13.sh`). Sizes are checked.
- **Model:** the E128 Kaggle copy chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1 (136 files, 131.09 GB), per file from Kaggle into `/root/models/e128` (`dl_e128.sh`, two sequential streams, forward and reverse).
  - Every file's sha256 is checked against the manifest embedded in the notebook's cell 1b (`verify_e128.sh`); it equals pod 2's e128 manifest. The dataset's e043_manifest.json is the pruning manifest, not hashes.
  - It is the row-chunked copy cell 2r's loader exists for, so the HF repo is not used.
- **Disk:** the C1 copy `/root/models/c1` (123 GiB on disk) was deleted to make room; it is on HF and on Kaggle under arazzd.
- **Pod system:** pod 2's dpkg fix, so Manh's cell-1 `sudo dpkg -i cu13-install/*.deb` succeeds as on Kaggle's image (`c1text/fix_cu13_deps.sh`). The debs were installed once, then `apt-get install -f --no-install-recommends` ran from the Ubuntu sources only. dpkg pass 2 rc=0; nvcc 13.0.88.
- **Kernel venv:** a fresh Kaggle-like `/root/kenv` (pod 2's kenv_fresh_freeze.txt, 48 packages), reset from `/root/kenv_fresh` at setup. Manh's cells install the v9 stack into it, so the vLLM files are the wheel's originals and cell 2r patches them as on Kaggle. The papermill kernelspec py313b32m points to it.
- **Layout (`setup_b32mv3.sh`):** fresh /kaggle/working, /tmp/dsv41, /tmp/arc_vision and /dev/shm/wheels. The competition dir is the local data/ files, with the test file replaced by evalx2t_s2026 (240 tasks, 344 pairs).
- **Runner (`launch.sh`):** papermill in a clean `env -i` environment, KAGGLE_IS_COMPETITION_RERUN=1, PATH = /root/kenv/bin, /usr/local/cuda/bin, then the system dirs.

## Deviations from the exact notebook (monitoring and Kaggle-like VRAM only)
1. **GPU memory hold (`gpu_hold.py`, started by `hold_trigger.sh` when cell 2 starts, i.e. after 2r and before cell 3 starts the server; left running all run):** one process holds **410 MiB per GPU**.
   - That is its CUDA context (186 MiB, libcudart via ctypes, no torch) plus one cudaMalloc, measured as nvidia-smi compute-app memory.
   - Why, from the memory auditor: on Kaggle, vLLM's allocator cap minus reserved was 0.67-0.73 GiB, against 1.16-1.22 on the earlier pod replica. That is ~0.38 GiB more non-torch memory on Kaggle (CUDA graphs 0.65 vs 0.34 GiB), and the replica also ran util 0.938 instead of 0.945.
   - Acceptance: vLLM's "Capped the PyTorch allocator at X GiB (Y GiB reserved)" gives X - Y = 0.67-0.73 GiB, with ~1.2 GiB free per GPU. Its startup check needs util x total free; 0.945 x 22.49 GiB leaves ~0.78 GiB of the pod's 22.04 GiB free, so the hold fits.
   - The solver's stop_server waits for nvidia-smi memory.used < 1,500 MiB per GPU, so the 410 MiB hold does not block its restarts.
2. **`ARC_SERVER_ENV_JSON`:** the notebook's own value, with `VLLM_GPU_MEMORY_LOG_INTERVAL` 60 instead of 300, so the server logs "GPU memory: …" per rank every 60 s. The settings cell uses `os.environ.setdefault`, so the env value wins. Logging only.

## Monitoring
- **`mon_sampler.py`** writes `mon.jsonl`. It samples every 60 s, and every 5 s while the solver's newest server event is "server died".
  - Host: /proc/meminfo (MemTotal, MemAvailable, AnonPages, Shmem, Cached, Mlocked, PageTables, AnonHugePages) and /dev/shm used.
  - Processes: VmRSS / RssAnon / RssShmem summed by group (vLLM workers, EngineCore, API server, solver, sandboxes, kernel, other); every vLLM process's RssAnon / RssShmem; the sandbox count, total and largest; the EngineCore count (two engines' shmem must never coexist).
  - GPU: nvidia-smi per GPU (used, free, power, SM clock, temp) and the compute apps.
  - **Kaggle-equivalent host RAM = AnonPages + Shmem + 7 GiB**, against Kaggle's cgroup limit of 173.0 GiB with no swap.
- **Other logs:** `mem.log` (60 s), `metrics_scrape.txt` (the server's /metrics every 60 s, across restarts), `gpu_hold.log`, and the server logs' "GPU memory:" lines.
- **Alerts (local `watch_b32.sh`, every 3 min; anything flagged goes to the coordinator at once):**
  - host RAM: Kaggle-eq >= 155 GiB WARN, >= 165 ALERT, >= 173 CRIT;
  - GPU: free < 1.0 GiB WARN, < 0.5 GiB ALERT;
  - any OOM / EngineDead / WorkerProc exception / execute_model timeout / Traceback line, a server death or restart, a burst of >= 5 error records in ~9 min, or the notebook ending.
  - The auditor's extra alerts are checked at each report: cap - peak < 0.26 GiB, and any increase in allocator retries.
- **Reports:** `pull_b32.sh` (copies the data) runs `b32_report.py` (RAM / VRAM trend table, cap and GPU memory per rank, deaths, errors) and `windows.py`.

## Timeline
| UTC | event |
|---|---|
| 01:43-01:56 | inputs prepared: C1 deleted; cu13-install v2 107/107; dpkg fix (pass 2 rc=0); fresh kenv; v9 wheel tree |
| 01:46- | E128 download (two streams; ~18-20 MiB/s) |
| 03:13:03 | E128 download done: both streams `DL_DONE ok=136 failed=0` (4,939 s and 4,553 s; ~27 MB/s combined) |
| 03:13:19 | `VERIFY OK: 136/136 files match cell 1b's manifest` |
| 03:13:21 | setup: archived c1text commit attempt 1's /kaggle/working (`/root/e066/c1commit1_working`); kenv reset (48 packages); notebook sha 771da13fd94d3746 |
| 03:13:21 | **launch** (rerun mode, `launch.sh 410`): papermill, mon_sampler, mem.log, scraper, holder trigger |
| 03:14:31 | try 1: hold started at **410 MiB/GPU** (context 186 MiB); cell 3 started 03:14:40. Install cells: "CUDA installation finished"; "B32-M ROW-CHUNK PATCHED" (ead1e087 -> b2a04fc2); E046 CHECK OK (solver 77fe80a3) |
| 03:28:02 | try 1: server up after 800 s. **GPU KV cache size 3,196,704 tokens** (Kaggle 3,174,039, +0.7%); Available KV 3.63 GiB; CUDA graph pool 0.34 GiB (Kaggle 0.65) |
| 03:27:58 | try 1 allocator caps, as cap / reserved (cap - reserved): TP0 20.37 / 19.71 (0.66); TP1 20.35 / 19.71 (0.64); TP2 and TP3 20.32 / 19.71 (0.61). Free 1.10-1.16 GiB. Kaggle: 20.42 / 20.36, reserved 19.69, **0.67-0.73**, free 1.17-1.23. So 410 MiB held ~60 MiB too much |
| 03:30:01 | try 1 stopped by PGID before any attempt finished (logs in `/root/e066/b32mv3/try1/`, `/root/e066/b32try1_working`) |
| 03:30:33 | **try 2 launched with a 350 MiB/GPU hold** (context 186 + 164 MiB malloc); fresh kenv and /kaggle/working |
| 03:31:42 | try 2: hold 350 MiB/GPU (context 186 MiB); cell 3 started 03:31:52 |
| 03:45:04 | try 2 allocator caps, as cap / reserved (cap - reserved): **TP0 20.43 / 19.71 (0.72); TP1 20.41 (0.70); TP2 and TP3 20.37 (0.66)**. Free 1.16-1.22 GiB. This matches Kaggle (0.73 TP0 .. 0.67 TP2/3; free 1.17-1.23) within 0.01 GiB. First GPU memory line: 19.32 allocated (peak 19.42), 19.71 reserved (0.27 split), 0 retries, 0 OOMs |
| 03:45:08 | try 2: **server up after 795 s**; KV 3,196,704 tokens; board free 628-686 MiB. Kaggle clock T0 ~03:30:40. The user's guidance: keep try 2 running whatever the cap, no more calibration |
| 03:50 | **The RAM WARN (160.2 GiB) was a measurement artifact.** This pod's /proc/meminfo is host-wide: 12-13 GiB of AnonPages belong to no process of the container, also at idle. The sampler's `kaggle_eq_gib` now uses the container's cgroup v1 memory.stat rss + shmem + 7 GiB (148.3 GiB at 03:50); `kaggle_eq_meminfo_gib` keeps the old formula. nvidia-smi board free of 626-686 MiB is the steady state, because vLLM caps its allocator at reserved + free - 0.5 GiB |
| 04:20 | status (0.82 h): stable. cap - peak TP0 0.58, TP1 0.56, TP2 and TP3 0.52 (0.61 / 0.59 / 0.55 / 0.55 after the warm-up at 03:46); peak 19.85, reserved 20.30, +0.07 GiB/h, so 0.26 is ~3.6 h away (~07:55 UTC). 0 retries, 0 OOMs. Kaggle-eq 149.2 GiB (+1.6 GiB/h, sandboxes). 15 first attempts, 11 correct; 0 errors; 453 gen tok/s; running 80, waiting 0 |
| 04:24 | tighter watch, per the coordinator: `watch2.py` (local, 60 s rounds, heartbeat `watch2.heartbeat`, alert state `watch2_state.json`) on `state_now.py` (pod). The sampler goes to 5 s via `fast.flag` once cap - peak < 0.35 or on the first retry. Alerts: retries, OOM, engine death or restart (logs preserved under /root/e066/b32mv3/preserved/), cap - peak < 0.40 / 0.30 / 0.26 or a drop > 0.1 in 30 min, Kaggle-eq > 155 or > 2 GiB/h, errored attempts / timeouts / HTTP 500, gen 0 for > 3 min, waiting > 30, server ERROR / Traceback, notebook end. Short reports every 30 min, full ones hourly |

## Reports
Dashboard data in `data/`, refreshed every 30 min by `export_dash.sh`:
- `gpumem.log`: the original "Capped the PyTorch allocator", "GPU memory:" and (from 04:37 UTC) "Avg generation throughput" lines of server.log, then of any server_restart*.log, in order.
- `ram.tsv`: utc, kaggle_clock_h, kaggle_eq_gib, 1 point per 10 min.

**Kaggle 1.0 h (04:31 UTC): stable, VRAM peak growing slowly; 0.26 is ~3.8 h away.**

| Kaggle h | peak | reserved | cap - peak TP0 / TP1 / TP2 / TP3 | retries | OOMs |
|---|---|---|---|---|---|
| 0.24 (server up) | 19.42 | 19.71 | 1.01 / 0.99 / 0.95 / 0.95 | 0 | 0 |
| 0.34 (after warm-up) | 19.82 | 20.26 | 0.61 / 0.59 / 0.55 / 0.55 | 0 | 0 |
| 0.83 | 19.85 | 20.30 | 0.58 / 0.56 / 0.52 / 0.52 | 0 | 0 |
| 1.00 | 19.86 | ~20.30 | 0.57 / 0.55 / 0.51 / 0.51 | 0 | 0 |

- VRAM: peak and reserved +0.066 GiB/h after the warm-up, the same on all ranks.
- Host: Kaggle-eq 148.8 -> 149.4 GiB (+1.3 GiB/h, all of it sandboxes 3.5 -> 4.0 GiB). Shmem flat at 125.1 GiB; one engine.
- Health: 0 errors / timeouts / HTTP 500. Running 80, waiting 0, KV 0.31, 0 preemptions; 466 gen tok/s, prefix hit 0.95.
- Accuracy: 19 first attempts, 15 fully correct; 22 of 344 pairs answered.

**First allocator retry at Kaggle ~1.26 h (04:46 UTC), on all 4 ranks at once.** The auditor's earlier pod replica first retried at 2.5 h, on TP2/3 only.
- 04:45:20: every rank peak 19.87, reserved 20.30 (0.26 split), 0 retries; free 0.61 (TP0/1) / 0.55 (TP2/3).
- 04:46:08: gen throughput 237.8 tok/s (from ~450-500), running 78 / waiting 1, KV 36.4%.
- 04:46:18: prefill burst, 1,206 prompt tok/s.
- 04:46:21, every rank: 19.34 allocated (peak 19.90), 20.32 reserved, 0.26 split, 1 retry, 0 OOMs.
  - cap - peak: TP0 0.53, TP1 0.51, TP2 and TP3 0.47.
- 04:46:28 on: back to 413-495 gen tok/s, running 79-80, waiting 0.
- The sampler switched to 5 s (`fast.flag`). Repeat retry alerts are now sent at most every 10 min.

**Kaggle 1.5 h (05:01 UTC): stable but tightening.** Allocator retries are climbing (TP0 / TP1 / TP2 / TP3 = 5 / 6 / 9 / 9 by 05:00, from 1 at 04:46; 3-4 at 04:51), 0 OOMs.
- cap - peak 0.52 / 0.50 / 0.46 / 0.46 GiB (peak 19.91): about -0.07 GiB/h on both hours, so 0.26 is ~2.7 h away (~07:45 UTC, Kaggle ~4.2 h).
- Kaggle-eq 149.7 GiB, flat. 405-484 gen tok/s, running 79-80, waiting 0. 0 errors, deaths or restarts.
- Accuracy: 40 first attempts, 34 fully correct; 49 of 344 pairs.
- Dashboard files now refresh every 5 min (`watch2.py` runs `export_dash.sh`; log `dash_sync.log`), including `data/status.json`.

**Kaggle 2.0 h (05:31 UTC): stable, VRAM growing faster; 0.26 is ~1.8 h away (~07:20 UTC, Kaggle ~3.8 h).**

| Kaggle h | peak | reserved (TP2/3) | cap - peak TP0 / TP1 / TP2 / TP3 | retries TP0 / TP1 / TP2 / TP3 | OOMs |
|---|---|---|---|---|---|
| 0.34 | 19.82 | 20.26 | 0.61 / 0.59 / 0.55 / 0.55 | 0 | 0 |
| 1.01 | 19.86 | 20.30 | 0.57 / 0.55 / 0.51 / 0.51 | 0 | 0 |
| 1.51 | 19.91 | 20.32 | 0.52 / 0.50 / 0.46 / 0.46 | 5 / 6 / 9 / 9 | 0 |
| 2.00 | 19.95 | 20.34 | 0.48 / 0.46 / 0.42 / 0.42 | 15 / 15 / 34 / 34 | 0 |

- VRAM: peak +0.091 GiB/h over the last hour (+0.066 the hour before). Retries come in storms (TP2/3 9 -> 23 within 05:01-05:11).
- Retries cost no visible throughput: 430-461 gen/s in the 30-min windows, prefix hit 95%, running 80, waiting 0, KV 0.40.
- Host: Kaggle-eq 149.4 -> 149.8 GiB (+0.5 GiB/h); shmem flat at 125.0 GiB.
- Health: 0 errors, timeouts, HTTP 500 or deaths.
- Accuracy: 62 first attempts, 52 fully correct; 82 of 344 pairs.

### CUDA OOM on TP3 at Kaggle 2.11 h (05:37:24 UTC)
The OOM came earlier than the auditor's 3.1-4.4 h band.
- `server.log` line 7437 (TP3 pid 68334, multiproc_executor.py:1189; execute_model -> all_gather -> torch.empty): `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 3 has a total capacity of 22.04 GiB of which 537.94 MiB is free. Process 66985 has 350.00 MiB memory in use. Including non-PyTorch memory, this process has 21.03 GiB memory in use. 20.37 GiB allowed; Of the allocated memory 20.06 GiB is allocated by PyTorch, ...`
  - The cap (20.37 GiB on TP3) was hit, not the device: 538 MiB was still free.
  - Process 66985 is the 350 MiB hold (the Kaggle-like non-PyTorch memory).
- Before it: cap - peak on TP2/3 0.42 (05:32), then 0.40 (05:33:37); retries 37 -> 41 (TP2/3); 446-495 gen tok/s with 80 running.
- 05:37:29: 198 gen tok/s. **05:37:39 on: 0 gen and 0 prompt tokens/s with 80 requests "running".** The engine hangs, as the auditor predicted for a non-output rank.
- Server logs copied to `/root/e066/b32mv3/preserved/053753/`.
- No intervention: B32's own restart path plays out.
- **05:42:27:** `EngineDeadError: EngineCore encountered an issue` (server.log 47035), **303 s after the OOM**. Rank 0 waited out VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (300 s).
- **05:42:29:** all **80 in-flight attempts** logged `retry <task>.0#0: TurnError: APIError: The server had an error while processing your request.`
  - On this V3 solver (77fe80a3) that is the HTTP 500 path, so each attempt used one of its 2 ARC_JOB_TRIES: 80 "retry" lines, 0 free re-queues, 0 attempts finalized as stop=error.
  - A second death would finalize the attempts that are in flight again.
  - Then `server died (code 0)`.
- **05:43:01:** `server start (restart 1)`, 32 s after the death; log `server_restart1.log`. Before the relaunch, the host had freed everything: Kaggle-eq 8.3 GiB, shmem 0, /dev/shm 0, 0 engines; on the GPUs only the 350 MiB hold. Two engines never coexisted.
- Logs preserved in `/root/e066/b32mv3/preserved/053753/` (OOM) and `/054231/` (death).
- Correction: **TP2 and TP3 both OOMed at 05:37:24.** They are the two ranks with the lowest cap (20.37 GiB); each logged `torch.OutOfMemoryError` twice.
- **05:45:41:** `server up after 160s` (restart 1), with KV 3,196,704 and the same caps (20.43 / 20.41 / 20.37 / 20.37, reserved 19.71). First GPU memory line: peak 19.42, 0 retries, free 1.16-1.22.
  - **Outage from OOM to serving: 8 min 17 s** (hang 303 s, death to relaunch 32 s, load 160 s with hot page and compile caches; Kaggle's load would be slower).
  - 1 free re-queue at server up (`ConnectError: Connection refused`).
  - 0 attempts finalized as stop=error. 81 error records (80 APIError 500 + 1 ConnectError).
- Dashboard fixes (coordinator):
  - `gpumem.log` sync: the remote grep loop ended on a no-match in server_restart1.log, so ssh returned 1 and the file was never renamed. The loop now ends in `; true`.
  - New `data/tps.tsv`: one row per minute from /metrics deltas, with 0 / 0 / 0 and an empty kv_pct while the server is down.
  - `status.json` counters are now cumulative over all server logs, with a floor file so they never go down.

**Kaggle 2.5 h (06:01 UTC): one blow-up (OOM 05:37 -> serving 05:45:41); cycle 2 is serving normally** (446-503 gen tok/s, running 79-80, waiting 0).
- New server: cap - peak 0.61 / 0.59 / 0.55 / 0.55 and cap - reserved 0.17 / 0.15 / 0.11 / 0.11. Peak 19.82 and reserved 20.26 are flat after the warm-up; 0 retries.
- **Lesson from cycle 1: the OOM came at cap - peak 0.40, not 0.26; the binding margin is cap - reserved.**
  - With expandable segments, reserved crept 20.26 -> 20.34 on TP2/3 (cap 20.37, so 0.03 left) over ~1.8 h, and then a 32 MiB all_gather buffer failed.
  - At cycle 1's ~0.045 GiB/h, the new server's 0.11 lasts ~2 h: the next OOM is expected ~07:20-07:50 UTC (Kaggle ~3.8-4.3 h).
- **Restart cost (cycle 1):**
  - hang 303 s + 32 s to relaunch + 160 s load;
  - then ~7 min of slow re-prefill (19-41 gen tok/s at 05:46-05:52) while the 80 retried attempts resubmitted;
  - about 15 min of serving lost, plus the progress of the 80 in-flight attempts.
- Accuracy: 63 first attempts, 53 fully correct; 104 of 344 pairs answered (including kept session answers).
- Counters: ooms 2, deaths 1, restarts 1, errored 81, retries_max 41.
- Watcher fix: the cap-peak drop alert now compares only samples of the current server after its warm-up (it had fired twice on the restart's ~0.4 GiB warm-up step).

**Kaggle 3.0 h (06:31 UTC): cycle 2 is retracing cycle 1 about 15-20 min faster; next OOM expected ~07:15-07:25 UTC (Kaggle ~3.75-3.9 h).**

TP2 (the tightest rank, = TP3) by server age; peak / reserved / retries:

| server age | cycle 1 | cycle 2 |
|---|---|---|
| 0 (up) | 03:45 19.42 / 19.71 / 0 | 05:45 19.42 / 19.71 / 0 |
| ~10 min | 03:55 19.82 / 20.26 / 0 | 05:55 19.82 / 20.26 / 0 |
| ~40 min | 04:25 19.86 / 20.30 / 0 | 06:25 19.87 / 20.28 / 3 |
| ~60 min | 04:45 19.87 / 20.30 / 1 | 06:30 retries 3, cap - reserved 0.05 |
| ~80-110 min | 05:05-05:35 19.93 -> 19.97 / 20.34-20.36 -> 20.22 / 11 -> 41 | |
| 112 min | 05:37 OOM (TP2 + TP3) | |

- The predictor is peak + the unreleasable split (0.26) against the cap (20.37). At the OOM, PyTorch had 20.06 allocated + 0.26 split + the 32 MiB request, about the cap.
  - Cycle 2 at 06:25: 19.87 + 0.26 = 20.13, so 0.24 GiB is left, and cycle 1 needed ~0.20 more to OOM.
- Reserved is not monotone: allocator retries flush the cache (20.36 -> 20.22 in cycle 1).
- Host: Kaggle-eq 147.3 GiB, flat; one engine.
- Health: no new errors since the restart; 393-511 gen tok/s, running 80, waiting 0.
- Accuracy: 67 first attempts, 57 fully correct; 109 of 344 pairs answered.

**Kaggle 3.0-3.5 h (06:31-07:01 UTC): cycle 2 at the edge, no OOM yet.**
- 06:39: cap - reserved 0.03 on TP2/3, the cycle-1 OOM level; peak step to cap - peak 0.46.
- 06:42: retry storm, retries 3 / 10 / 11 / 11 -> 8 / 17 / 22 / 22; cap - peak 0.43 on TP2/3 (peak 19.94).
- 06:43-07:01: a plateau. Peak 19.94, reserved 20.36 (cap - reserved 0.01), split 0.26, free 0.51, retries 27 on TP2/3, 0 OOMs. The retries flush the cache and keep it alive. Cycle 1 OOMed 3-12 min after the same peak.
- Serving: 400-491 gen tok/s, running 79-80, waiting 0. Kaggle-eq 147.5 GiB.
- Accuracy: 76 first attempts, 63 fully correct; 119 of 344 pairs.

**Kaggle 4.0 h (07:31 UTC): cycle 2 has been at the edge for ~50 min, held up by retry storms; no OOM yet at 1.77 h of server age** (cycle 1 died at 1.87 h).

TP2 (= TP3), cycle 2:

| UTC (age) | peak | reserved | cap - reserved | retries |
|---|---|---|---|---|
| 05:55 (0.2 h) | 19.82 | 20.26 | 0.11 | 0 |
| 06:25 (0.7 h) | 19.87 | 20.28 | 0.09 | 3 |
| 06:45 (1.0 h) | 19.94 | 20.34 | 0.03 | 25 |
| 07:15 (1.5 h) | 19.94 | 20.36 | 0.01 | 32 |
| 07:31 (1.77 h) | 19.96 | 20.22 | 0.15 | 75 |

- Retries are accelerating (TP2/3 27 -> 75 in 30 min; TP1 61, TP0 38). Each one flushes the cache, so reserved swings 20.22-20.40; TP0 is at 20.40 of its 20.43 cap.
- cap - peak 0.47 / 0.45 / 0.41 / 0.41 (peak 19.96 vs cycle 1's logged 19.97 before its OOM).
- Serving: 459 gen tok/s mean over the hour (387-511), running 79-80, waiting 0. 0 new errors.
- Host: Kaggle-eq 147.7 GiB (+0.4 GiB/h); shmem 125.1; one engine.
- Accuracy: 87 first attempts, 73 fully correct; 131 of 344 pairs answered.

### Second OOM (TP2 + TP3) at Kaggle 4.02 h (07:32:01 UTC), 1.77 h into cycle 2
- `server_restart1.log`, TP3 pid 86567, the same all_gather -> torch.empty path: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 3 has a total capacity of 22.04 GiB of which 515.94 MiB is free. Process 66985 has 350.00 MiB memory in use. Including non-PyTorch memory, this process has 21.05 GiB memory in use. 20.37 GiB allowed; Of the allocated memory 20.11 GiB is allocated by PyTorch, with 126.00 MiB allocated in private pools (e.g., CUDA Graphs), and 265.63 MiB is reserved by PyTorch but ...`
- TP2 logged the same; 2 lines each.
- Just before (07:31): peak 19.96 on every rank; reserved 20.22 (TP2/3) / 20.40 (TP0); retries 75 / 75 / 61 / 38; gen 471-527 tok/s until 07:32:01, then 0.
- At the failure: 20.11 GiB allocated + 0.27 unreleasable reached the 20.37 cap. Cycle 1 failed at 20.06 allocated.
- Logs preserved in `/root/e066/b32mv3/preserved/073259/`.
- The attempts still in flight that already used a try at the cycle-1 death have none left and will be finalized as errors at this death.
- **07:37:03:** `EngineDeadError` (server_restart1.log), 302 s after the OOM.
- **07:37:00-07:37:15:** the 80 in-flight attempts got the HTTP 500.
  - **56 were finalized as `stop=error`.** These were retried at the cycle-1 death, and their last try is now used. They had been running since 05:45 (1.85 h), e.g. `[89] 25acf30e.0#0 pred=no stop=error turns=8 out=7088 ... 6678.7s`.
  - The other 24 were retried once.
- **07:37:15:** `server died (code 0)`.
- **07:37:40:** `server start (restart 2)`, log `server_restart2.log`. Host Kaggle-eq was back to 9.5 GiB before the relaunch.
- Counters: 248 attempt records, 160 with error; deaths 2, restarts 2 (the solver allows 3), OOM events 2.
- Logs preserved in `/root/e066/b32mv3/preserved/073259/` and `/073758/`.
- **07:40:15:** `server up after 155s` (restart 2); KV 3,196,704, the same caps (20.43 / 20.41 / 20.37 / 20.37). 1 free re-queue at up.
  - **Outage 2, OOM to serving: 8 min 14 s.** Cycle 3 started.
  - At the cycle-1/2 rate, the next OOM comes ~09:25 UTC (Kaggle ~5.9 h), and there is 1 restart left after it.

**Kaggle 4.5 h (08:01 UTC): two blow-ups (05:37 and 07:32). Cycle 3 is serving normally and tightening ~10 min ahead of cycle 2; next OOM expected ~09:10-09:25 UTC (Kaggle ~5.7-5.9 h).**
- Recovery after restart 2 took 12 min (07:40 up -> 07:52 at 80 running / 0 waiting): all 80 attempts restarted on a cold prefix cache (hit 4.5%) and were prefilled at 1,000-2,300 prompt tok/s. After restart 1 it took ~7 min.
- 08:01, TP2/3: peak 19.86, reserved 20.30, cap - reserved 0.07, 0 retries. cap - peak 0.57 / 0.55 / 0.51 / 0.51; cycle 2 at the same age had 0.55 on TP2/3.
- Serving: 401-458 gen tok/s, running 80, waiting 0. Kaggle-eq 147.1 GiB.
- Losses: 2 deaths, 2 restarts (1 left); 56 attempts finalized as errors; 162 error records.
- Accuracy: 88 first attempts, 74 fully correct; 141 of 344 pairs answered.
- Fix: the 5-min dashboard sync now runs before any watcher exit. It had been skipped when the watcher exited at a report mark.

**Kaggle 5.0 h (08:31 UTC): two blow-ups; cycle 3 at the edge (cap - reserved 0.03 on TP2/3 since 08:24). Next OOM likely ~09:10-09:30 UTC; one restart left after it.**

| cycle | server up | first retry (age) | cap - reserved 0.03 (age) | OOM (age) |
|---|---|---|---|---|
| 1 | 03:45 | 04:46 (1.0 h) | ~05:31 (1.8 h) | 05:37 (1.87 h) |
| 2 | 05:45 | 06:22 (0.6 h) | 06:39 (0.9 h) | 07:32 (1.77 h) |
| 3 | 07:40 | 08:20 (0.67 h) | 08:24 (0.73 h) | - |

- Gen tok/s by Kaggle hour: h0 408, h1 446, h2 352, h3 459, h4 294. The dips are the two outages; 36 min in total below 50 tok/s.
- Restart cost per cycle: ~5 min hang + ~30 s + ~2.6 min load, then 7-12 min of cold re-prefill, so about 15-20 min of serving. In-flight attempts are retried once and lost at the next death (56 at death 2).
- Accuracy: 98 first attempts, 84 fully correct; 152 of 344 pairs answered; 162 error records.
- Projection:
  - An OOM ~09:20 uses the last restart (3 of 3).
  - The next one (~11:15 UTC, Kaggle ~7.7 h) leaves no server for the last ~4 h.
  - On Kaggle each reload is 19-30 min, so the same cycle there loses ~1 h of serving per OOM.

**Accuracy trend (coordinator's method, `dashboard/tools/acc_series.py`; helper `pod3_scripts/b32mv3/acc_report.py`):** the ARC task score so far over all 240 tasks (per task, the fraction of test pairs right by any finished attempt; unanswered = 0), plus the tasks answered and the first-attempt accuracy (fully correct / first attempts with an answer, including kept session answers).
- At Kaggle 5.23 h: **score 40.4**, 120 of 240 tasks answered, first-attempt accuracy 74.6%.
- Score at the end of each Kaggle hour (tasks answered):

  | h1 | h2 | h3 | h4 | h5 | h6 so far |
  |---|---|---|---|---|---|
  | 6.2 (19) | 21.9 (62) | 25.0 (77) | 32.1 (95) | 37.3 (111) | 40.4 (120) |

  The slow steps (h3 +3.1, h5 +5.2) are the hours of the two outages.
