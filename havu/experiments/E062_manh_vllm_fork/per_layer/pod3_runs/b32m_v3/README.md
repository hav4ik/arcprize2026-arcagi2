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
