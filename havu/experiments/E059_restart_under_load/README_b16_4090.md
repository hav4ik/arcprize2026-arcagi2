# E059 B16 on 4x RTX 4090: restart under load at 16 GiB KV offload

Status (2026-10-27 19:35 UTC): **done. Verdict: PASS.** The restart peak is 0.98x the steady anon+shmem (the rule was
<= ~1.1x). The solver detected the death, restarted the server once and lost no job tries.

| | result |
|---|---|
| container anon+shmem: steady before the kill / peak during the restart | **124.1 / 121.4 GiB (0.98x)** |
| old server memory gone / new server allocating / server up again | +7 s / +45 s / **+125 s** after the kill |
| server death detected / restarts | +2 s / **1** |
| in-flight attempts lost to the kill | 38 jobs, all re-queued as server-side errors (S15): **0 job tries spent, 0 jobs ended with an error** |
| gen tok/s: 30 min before the kill / kill+5..10 / kill+10..15 / kill+25..30 min | 627 / 596 / 833 / 871 |
| first 5-min window at >= 90% of pre-kill | **kill + 9.8 min** (new server up + 7.8 min) |
| after the restart, up to the force deadline | 464 preemptions, KV > 0.95 for 51% of the time; host pool absorbed it: true uncached 2.4%, 741 gen tok/s, in-flight context peak 3.33M (cliff region 4.5-4.8M not reached) |
| local score (one draw) | **102/130 pairs** |

Twin: the 32 GiB L4 run (`README.md`, `launch_b32kill.sh`) also passed, at a ratio of 1.05.

## Question

Candidate "B16" is the configuration of the Kaggle-smoked notebook:
- candidate B: venv_candb = kit overlay + E054 indexer skip + E051 KV offload; pinned_candb solver `77fe80a3`;
- 56 workers, KV gate 0.85 with the queue hold;
- 16 GiB of private pinned KV offload.

If the vLLM engine dies mid-run, does the solver's restart path hold up?
- **Memory.** The old server's Engram (~91 GiB) and pinned pool (16 GiB) must be released before the new server
  allocates its own. Otherwise host memory briefly doubles, which would be fatal under Kaggle's 173 GiB cgroup.
- **Jobs.** The solver must detect the death, restart the server and re-queue the lost attempts without spending their
  job tries (S15).
- **Throughput.** It must recover.

**Design.** This is the 4090 twin of the L4 run `launch_b32kill.sh` (32 GiB). `killer.sh` SIGKILLs this run's own
`VLLM::Worker_TP1` 1.5 h after "server up". It finds the worker among the solver's descendants, never by name. It also
samples host memory every 2 s from 5 min before the kill to 20 min after.

**Pass rule** (set before the run): the peak anon+shmem during the restart is <= ~1.1x the steady value before the kill.

## Box and setup

- **Box.** Vast 4x RTX 4090, instance **57983124**: Virginia, machine 54992, image `vastai/vllm:v0.32.0-cuda-13.0`,
  driver 580.126.09, kernel 6.8.
  - Direct ssh: `ssh -p 62174 root@159.112.204.189`. The Vast proxy was not used.
  - GPUs: 24,564 MiB each, **power limit 300 W** (max 530 W; E057b's box ran at 450 W). PCIe Gen4 x16, one NUMA node
    (PHB within pairs 0-1 and 2-3, NODE between them).
- **CPU.** Threadripper PRO 5975WX, `nproc` 64, but the container's `cpu.max` is `2048000 100000`: **only ~20.5
  effective CPUs** (Vast lists 21.3).
  - The run used 5.5 cores on average.
  - It was throttled in 108 CFS periods, 324 s in total, mostly during the server's compile and start. The quota did
    not bind in steady state.
- **Memory.**
  - MemTotal **503.5 GiB**; cgroup v2 `memory.max` **279.1 GiB** (299,657,854,976 B).
  - `/dev/shm` is 83 GiB. It is not used: Engram and the offload pool are pinned allocations that `meminfo` counts as
    Shmem.
  - `/proc/meminfo` is host-wide. Other tenants held ~20-30 GiB of AnonPages throughout, so the container's own numbers
    (cgroup `memory.stat` anon + shmem) are the primary ones below.
  - The cgroup also counts page cache. After the model download and load, `memory.current` sat at 269-277 GiB (max
    277.2), near the limit, with ~155 GiB of it reclaimable page cache. Reclaim was never a problem: the new server's
    107 GiB allocation went through without stalls.
  - Half-RAM rule: one B server needs ~124 GiB, under half of 503.5 GiB and under the cgroup limit.
- **Disk.** 204 GB overlay; 166 GB used after setup.
- **Image services.** I ran `supervisorctl stop vllm model-ui`. As on earlier 4090 boxes, this left the image's Qwen
  vLLM orphans (`vllm serve Qwen/Qwen3.5-9B`, EngineCore, 4 workers; pgid 1174) holding 21.6 GB per GPU. I checked the
  group leader's command line, then sent SIGTERM to the group. GPUs went back to 1 MiB.

### Setup chain (`b16_4090/setup_chain.sh` + `go.sh`, on the box with `setsid`; log `results/setup/e059/runs/setup_chain.log`)

1. **Stack.** E042 `scripts/setup_box.sh` → `box_install.sh`: INSTALL OK in 65 s (16:12:56-16:14:01).
2. **Model download.** Plain HTTP from the start (`setup/dl_e128_http.sh` = E057b's script, `HF_HUB_DISABLE_XET=1`,
   resumable). The token came from `huggingface_full_access_api_key` over SSH stdin only.
   - The local watchdog (`setup/dl_watchdog.sh`) never had to act.
   - **DL_DONE in 14.4 min** (16:13:23-16:27:45, 240-255 MiB/s, then ~105): 121 files, **131,090,911,045 B**,
     revision `9bce98c5`.
3. **podprep.sh** (E046 pods): kit `a9706290` → `/root/kit` and its 12-file overlay over `/root/dsv41/venv`
   (sha-checked), toolenv, eval data. 4 s.
4. **E056 `setup_candb_venv.sh`** → `/root/dsv41/venv_candb`, 34 s.
   - `cp -a` with 116 shebangs rewritten.
   - The notebook-cell install of overlay `1a3a6679` / `624e7754`: **16/16 sha lines match**.
   - Source venv unchanged.
   - **E051 CPU tests 26/26 OK**.
5. **`/root/kit_b`** = `/root/kit` plus `kaggle/dsv41_tools/pinned_candb/` at kit paths: solver **77fe80a3**, harness
   **882c817f**, train_check **8fe36175**, auto_repair `2292369c`, names `571f47eb`.
   - Workload `evalx2t_s2026` in `/root/data`: challenges `83b249b1398d`, solutions `8ed7202a78eb`, heavy90
     `3c6db22a194e`.
6. **Memory check**: plan 130 GiB <= min(half of MemTotal, cgroup limit) = 251.8 GiB.
7. **One server start** (E056 `verify_candb_start.sh` → `serve_candb.sh`, 16 GiB, util 0.92, override 21490):
   - **ready after 230 s**; the offload line once (`swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped
     scratch groups [8]`);
   - `Overriding num_gpu_blocks=29122 with num_gpu_blocks_override=21490` → **GPU KV cache size 1,591,736 tokens**;
   - indexer skip on in all 6 processes; Engram 23.60 GiB pinned per rank;
   - **0 tracebacks** at ready and after the chat; the chat returned `42`, finish `stop`;
   - Shmem 0.1 → 107.1 GiB, server PSS 119.2 GiB;
   - stopped by PGID: GPUs back to 1 MiB. The 1 traceback after the stop is vLLM's usual `EngineDeadError` shutdown
     noise, as in E056 and E057b.
   - The chain gated on all of these before starting the run (`CHAIN OK in 861 s`).

**The E056 bundle was rebuilt from the repo** (the earlier boxes are gone). Every file matches the bundle shipped to
E057b's box except the scratch notebook.
- The notebook is not committed. `kaggle/dsv41_tools/build_frozen_mt.py`'s default harness no longer passes its import
  check (`arc_harness.grid_image`, E058), so I rebuilt it with B's pinned files:
  `build_frozen_mt.py --slug e056-candb-installtest --workers 36 --solver/--harness/--train-check kaggle/dsv41_tools/pinned_candb/...
  --overlay experiments/E056_candidate_b/kit --retention 512 --server-extra '["--kv-transfer-config", <32 GiB JSON>]'`.
  The result is sha `dcc2a7a3` instead of `831cb9ca`.
- Cells 0, 1, 2, 3 and 4 are byte-identical to `831cb9ca`. Cells 00 and 0b differ only in the embedded pins and two
  newer printed settings.
- The overlay it installs is the same pair, and the 16/16 sha check covers it.

## Run (exact settings)

`b16_4090/run_b16.sh e059_b16kill 5400` (E057b's `run_arm.sh` adapted). It runs the real solver through
`/root/podrun.sh` + `/root/local_run.py`, starting at **T0 16:32:04 UTC**.
- **Code and venv:** `ARC_KIT=/root/kit_b`, `ARC_VENV=/root/dsv41/venv_candb`,
  `PATH=/root/dsv41/venv_candb/bin:/usr/local/cuda-13.0/bin:...`.
- **Solver settings:** `ARC_WORKERS=56 ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 ARC_CONTEXT_LIMIT=163840
  ARC_KV_ADMIT_MAX=0.85 ARC_KV_ADMIT_MIN_WORKERS=16 ARC_ADMIT_MAX_WAITING=1 ARC_AUTO_REPAIR=1 ARC_MAX_NUM_SEQS=64
  ARC_GPU_UTIL=0.92 ARC_CACHE_SALT=e059_b16kill_163204`.
- **Workload:** `ARC_CHALLENGES/SOLUTIONS` = `evalx2t_s2026`, with `ARC_TASK_IDS` = the 90 heaviest tasks (130 pairs),
  the same list as the L4 twin.
- **Deadlines:** `ARC_STOP_NEW_S=9600 ARC_FORCE_AT_S=10200 ARC_HARD_STOP_S=10500 ARC_KILL_AT_S=10800`.
- **`ARC_SERVER_EXTRA_ARGS`:** Humming, FULL graphs cap 64, `--prefix-cache-retention-interval 512`, and
  `--kv-transfer-config` OffloadingConnector with `cpu_bytes_to_use` **17179869184** (private, checkpoint, lru), plus
  `--num-gpu-blocks-override 21490`.
- **Environment:** the image's credentials and its Qwen `VLLM_*` variables are unset before the launch.
- **Kill:** `killer.sh <run> 5400` (the repo version with the `$NF` fallback, `434f0286`).
  - The run script also writes `solver.pid`, which `killer.sh` reads first.
  - Before the kill, a read-only dry run of its descendant walk found `VLLM::Worker_TP1` (pid 12049) under solver
    11080 in server group 11157.
- **Samplers:**
  - `sampler.py` every 10 s;
  - `mem2s_guard.log` every 2 s: host AnonPages/Shmem plus the cgroup's anon/shmem/file and `memory.current`, all run;
  - memdetail every 30 s: also per-process PSS and `cpu.stat`;
  - `nvidia-smi` every 30 s;
  - `killer.sh`'s `mem2s.log`: host meminfo, VLLM process count and GPU MiB every 2 s around the kill.
- **Memory guard** (protects the box): SIGTERM to the solver's process group if host AnonPages + Shmem > 85% of the
  cgroup limit (237 GiB) in 2 samples in a row, 2 s apart. **It never fired**: the host-wide peak was 153 GiB.
- **End:** every server process group seen is recorded. At the end, both were gone and the GPUs were at 1 MiB.

## Results

Analysis: `b16_4090/analyze_b16.py` (restart-aware; it reuses E057b `compare_arms.py`) →
`results/analysis_b16kill.{json,md}`. Times are UTC. "Kill" = 18:06:05 = T0 + 5641 s.

### 1. Memory at the restart: PASS (0.98x)

| (GiB) | steady: 5 min before the kill | restart window: kill → +20 min | peak / steady |
|---|---|---|---|
| **container anon + shmem** (cgroup) | median **124.1** (123.8-124.3) | **max 121.4**; min 1.9 at +7 s | **0.98** |
| container shmem (Engram + pinned pool) | 106.96 | old 106.96 until +1 s; new server exactly 106.95 | 1.00 |
| container anon | 17.1 | max 14.5 | 0.85 |
| host AnonPages + Shmem (with other tenants) | 145.9 | max 143.3 | 0.98 |

No 2-s sample in the window had shmem above the steady value: the old and new servers' memory never overlapped. The
timeline below is from the 2-s logs and `solver.log`.

| UTC | after the kill | event |
|---|---|---|
| 18:06:05 | 0 s | `killer.sh`: SIGKILL `VLLM::Worker_TP1` pid 12049 (solver 11080). Before it: 597-627 gen tok/s, KV 0.83-0.87, 38 attempts running (gate holding) |
| 18:06:05-07 | 0-2 s | all 38 in-flight attempts get "EngineCore encountered an issue" (37 as APIError, 1 as HTTP 500) |
| 18:06:07 | +2 s | the API server exits (code 0); the solver logs **"server died"** |
| 18:06:07-12 | +2-7 s | old memory released: container shmem 107.0 → 42.6 → **0 GiB**, anon 17.7 → 1.9 GiB |
| 18:06:22 | +17 s | all old vLLM processes gone and all 4 GPUs at 1 MiB (GPU0 by +7 s) |
| 18:06:25 | +20 s | **restart 1 starts**: the solver's `stop_server()` stops the old group and waits for free GPUs first |
| 18:06:50 | +45 s | the new server is allocating Engram (container shmem 12.5 GiB); 91.0 GiB at +67 s |
| 18:08:02-06 | +117-121 s | the new 16 GiB pinned offload pool is allocated (shmem 103.2 → 107.0 GiB) |
| 18:08:10 | **+125 s** | **"server up after 105 s"**: KV pool 1,591,736 tokens, unchanged (warm compile cache, model files in page cache) |
| 18:15:55 | +590 s | first trailing 5-min window at >= 90% of pre-kill gen tok/s |

The new server allocated only after the old memory was gone. Two things guarantee that order:
- the old engine's processes exit by themselves within seconds of the engine death;
- the solver's restart path stops the old process group and waits for the GPUs to free before starting the new server
  (S5).

### 2. Solver log: death detected, 1 restart, every lost attempt re-queued (S15)

- **Detection:** "server died" 2 s after the kill. 1 restart (`status.json` restarts = 1), no startup failure, and no
  "no restart left".
- **Jobs:** 38 jobs were in flight. They produced **49 error records**, and every one was re-queued through the
  server-side path ("re-queue ... after a server-side error"):
  - 37 `APIError: EngineCore encountered an issue`;
  - 2 `HTTP 500 EngineCore encountered an issue`;
  - 10 `ConnectError: Connection refused`.
- **Job tries:** **0 "retry" lines**, so no `ARC_JOB_TRIES` were spent.
- **Outcomes:** **0 jobs ended with an error.**
  - Of the 38 jobs, 22 later ended naturally and 16 were cut at the force deadline; 23 ended with an answer.
  - 4 had their session answer harvested and kept until the re-run.
- **Re-launch race (new finding, harmless here).** 37 of the 38 jobs were re-queued at 18:06:06-07, before the
  watchdog's "server died" line at 18:06:07. The 38th was processed after the restart.
  - In that second the server process was still alive and `server_ok` was still true, so the launch loop started 11
    of the re-queued jobs again against the dying server.
  - They failed after 0.3-0.4 s: 1 HTTP 500 at once, and 10 "Connection refused" that were processed after the
    restart returned.
  - Each spent a second `ARC_CONN_TRIES` re-queue (cap 4, separate from job tries).
  - With up to 3 restarts, a job could reach the cap and then spend job tries. A cheap fix: treat a server-side error
    record as `server_ok = False` until the server answers again.
- **Work lost:** the 38 killed attempts had generated **1.31M completion tokens**, discarded because re-runs start over.
  Some had been running for 90 min. That is ~35 min of pre-kill generation, much more than the 125-s outage.
- **Logging gap** (also found on the L4, fix-list S19): the solver's first `server:` line after the restart reports
  negative rates (−11,744 gen tok/s), because the server counters reset. It affects only `solver.log`,
  `metrics.jsonl` and `status.json`; nothing reads those values for decisions.

### 3. Throughput recovery

Generated tok/s comes from `/metrics` (10-s samples, counter reset removed), as 5-min windows relative to the kill:

| window | gen tok/s | vs 30 min before the kill |
|---|---|---|
| kill − 30 → kill | **627** (the last 15 min: 597) | 1.00 |
| kill → +5 min (125 s outage; the new server's first 5 min: 156) | 11 | 0.02 |
| +5 → +10 min | 596 | 0.95 |
| +10 → +15 min | 833 | 1.33 |
| +25 → +30 min | 871 | 1.39 |
| +55 → +60 min | 713 | 1.14 |

- **Recovery time:** the first trailing 5-min window at >= 90% came at **kill + 590 s** (new server + 465 s); 100% at
  kill + 620 s.
- **Why it then exceeds pre-kill:** before the kill the gate had shed concurrency to 28-38 long attempts at KV ~0.87.
  After the restart all contexts are empty, so the gate admits 56 fresh attempts at once, which decode faster.

### 4. Rest of the run (to the force deadline)

| | before the kill (server 1, up → kill, 90 min) | after the restart (server 2, up → force, 74 min) |
|---|---|---|
| gen tok/s | 662 | 741 |
| KV mean / peak | 0.759 / 0.975 | 0.793 / **1.000** |
| time with KV > 0.90 / > 0.95 | 21% / 0.2% | 54% / **51%** |
| **preemptions** | **0** | **464** (378/h; at +2.15-2.75 h) |
| server running mean; waiting > 0 share | 43.3; 12% | 43.4; **61%** (max 53) |
| solver running mean; gate held | 45.3; 56% | 52.2; 49% |
| GPU prefix-hit share | 0.835 | 0.334 |
| host-restored tok/s; host hits / queried | 1,658; 0.13 | **9,481**; 0.64 |
| **true uncached share** | 3.7% | **2.4%** |
| TTFT mean; p50 / p90 bucket | 5.8 s; 1.0 / 5.0 | 15.1 s; 2.5 / 40 |
| queue time mean | 4.8 s | 14.2 s |
| offload stored / loaded | 380 / 44 GiB | 341 / 249 GiB |
| **in-flight context** mean / peak (traces) | 1.91M / 2.46M | 2.22M / **3.33M** (at +2.60 h) |
| minutes with in-flight > 4.5M | 0 | 0 |

**A restart makes the next hour run hotter, and the host pool carried it.**
- **What happened:** the restart reset KV to ~0.3, and the gate admitted 56 attempts at once. These were mostly the
  re-queued, long-running attempts on hard tasks (the in-flight set at the kill is biased toward long attempts) plus
  the queue head. The cohort grew in step. From +2.15 h, KV sat at 0.94-0.99 (5-min means) with 4.7-10.7 requests
  waiting on average and 464 preemptions.
  - Before the kill, the first wave had shed concurrency gradually and never preempted.
  - E057b's 56-worker arm on the same workload also had 0 preemptions.
- **Why it cost little:** preempted and evicted prefixes came back from the 16 GiB host pool, at 12-20k restored
  tok/s in the hot phase. GPU prefix hits fell to ~0, but the true uncached share stayed at 2.4% and gen tok/s at
  680-830. The traces show 1 full re-prefill out of 1,033 turns >= 50k tokens, and 1.7% uncached prompt tokens after
  the first turn.
- **No host-pool cliff:**
  - In-flight context peaked at 3.33M, below the 4.5-4.8M region.
  - Host-hit ratio r over 180-s windows after the restart: the 13 qualifying windows after warm-up (waiting > 0 in at
    least half the samples) had **median r 0.991, none below 0.5**.
  - r < 0.5 occurred only in the first ~25 min after the restart: the host pool was empty, and queries were mostly
    first-turn prompts.
- **On Kaggle's L4s** (slower prefill, PCIe Gen3) the same preemption burst would cost more wall time than here.
  Consider ramping admissions after a restart, for example up to the pre-death running count, rather than refilling
  all 56 at once.

**Memory growth after the restart** (container anon+shmem, 2-s log):
- 120.95 GiB in the first 5 min after warm-up, 122.65 GiB in the last 5 min before the force deadline: **+1.6 GiB/h**.
- Peak 126.3 GiB. The host-wide peak, other tenants included, was 153.0 GiB. MemAvailable never went below 332 GiB.
- `memory.current` peaked at 277.2 GiB of 279.1 GiB. Almost all of it above the ~126 GiB of anon+shmem is
  reclaimable page cache.

**GPU:** utilization 90-91%, power 225-241 W (at the 300 W cap), memory peak 22.4-22.5 GB.

### 5. Local score (one draw)

**102/130 pairs** (`local score` in `solver.log`).
- 135 jobs finished: 112 ended naturally (105 submitted, 4 output budget, 3 max turns) and 23 were cut by the
  deadlines (22 open streams at the force deadline).
- 122 pairs had an answer.
- Admission closed at +2.52 h with 45 jobs unstarted (the min-request-tokens rule: 19 min × 13.2 tok/s per stream).
- Over the whole run, 164 pair-answers had 83.5% exact.

Report only: it is one draw over a heavy-first 3 h run that includes one engine death.

## Verdict

- **Memory: PASS.** The restart peak is 0.98x steady in the container (0.98x host-wide).
  - The old Engram + pool were freed 7 s after the kill. The new server started allocating at +45 s.
  - At 16 GiB, one engine death needs no extra host memory.
  - The L4 twin at 32 GiB measured 1.05x.
- **Restart path: PASS.**
  - Death detected in 2 s; 1 restart; up again in 125 s; KV pool unchanged.
  - All 38 lost attempts re-queued without spending job tries; 0 jobs ended with an error.
  - Two cheap fixes to consider: the re-launch race (above), and S19 for the negative rates after a restart.
- **Recovery: PASS.** >= 90% of pre-kill throughput 9.8 min after the kill.
  - The real cost of one death is ~2 min of outage plus the discarded partial work of the in-flight attempts: 1.31M
    tokens, ~35 min of generation.
- **After the restart:** hotter KV, 464 preemptions, no capacity cliff, and throughput held thanks to the host pool. A
  post-restart admission ramp is worth considering for the L4/Kaggle case.

## Caveats

- **One draw.** The stack is not run-to-run deterministic.
- **4090, not L4.**
  - Faster prefill and decode; GPUs capped at 300 W here.
  - The restart reloads from local disk with the model in page cache (105 s); Kaggle reloads from its mount (~10 min).
  - The KV pool is pinned to 1,591,736 tokens (+2.4% over Kaggle's).
- **One failure mode.** This was a SIGKILL of one TP worker. A CUDA error or an OOM kill of EngineCore may tear down
  differently. The release order seen here comes from the processes exiting and from the solver's group stop + GPU
  wait.
- **Host-wide meminfo includes other tenants** (~20-30 GiB of AnonPages); the cgroup numbers do not.
- **CPU and memory.** The tool sandboxes ran on a 20.5-CPU quota, which did not bind in steady state. The box's cgroup
  also holds the page cache of the 131 GB model, which is reclaimable.

## How to reproduce

```bash
# local: stack + model (token over SSH stdin) + bundles
experiments/E042_dsv41_l4_synthetic/scripts/setup_box.sh root@<ip> <port>
sed -n 's/^huggingface_full_access_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' " | ssh -p <port> root@<ip> \
  'read -r T; HF_TOKEN="$T" setsid nohup /root/dsv41/dl_e128_http.sh > /dev/null 2>&1 < /dev/null &'   # after copying setup/dl_e128_http.sh and creating /root/hfenv
# ship /root/podkit (kit_ds dsv41_kit.tgz a9706290, E046 pods podprep/podrun/memlog, kaggle/dsv41_tools/out/local_run.py, eval jsons),
# /root/e056 (the E056 bundle, rebuilt as above), /root/e059/{scripts (setup_chain.sh run_b16.sh go.sh ../killer.sh E057b sampler.py),
# workloads (evalx2t_s2026_*), pinned_candb (kaggle/dsv41_tools/pinned_candb/*)}; then on the box:
setsid -f bash /root/e059/scripts/go.sh > /root/e059/runs/go.driver.out 2>&1 < /dev/null   # setup chain, then the run
# local, after copying /root/runs/e059_b16kill back without scratch/ (secret-scanned):
.venv/bin/python experiments/E059_restart_under_load/b16_4090/analyze_b16.py \
  experiments/E059_restart_under_load/b16_4090/results/runs/e059_b16kill \
  --out experiments/E059_restart_under_load/b16_4090/results/analysis_b16kill.json --md experiments/E059_restart_under_load/b16_4090/results/analysis_b16kill.md
```

## Files (`b16_4090/`)

- **Scripts:**
  - `setup_chain.sh`: steps 1-8 with gates; `FROM=5` resumes at the download wait.
  - `run_b16.sh`: the run, samplers, memory guard, `solver.pid`, killer launch, cleanup by PGID.
  - `go.sh`: the chain, then the run.
  - `analyze_b16.py`.
  - `setup/dl_e128_http.sh`, `setup/dl_watchdog.sh` (local).
- **`results/runs/e059_b16kill/`:** the run dir without `scratch/`.
  - `run.log`, `env.txt`, `arm_t0`, `solver.pid`, `server_pgids.txt`, `killer.{log,out}`, `mem2s.log`,
    `mem2s_guard.log`, `memdetail.log`, `metrics10.jsonl`, `gpu.csv`.
  - `out/dsv41_logs/`: `solver.log`, `attempts.jsonl`, `metrics.jsonl`, `server.log`, `server_restart1.log`,
    `server_cmd.json`, `status.json`, `stage.log`.
  - `out/dsv41_traces.tar.gz` and `out/submission.json`.
- **`results/analysis_b16kill.{json,md}`.**
- **`results/setup/`:**
  - `e059/runs/`: setup chain log, podprep, run progress, drivers.
  - `e056/runs/`: `setup_candb_venv.log`, `notebook_install.log`, `e051_cpu_tests.log`, source-venv shas, and the
    verify start's progress, `server.log`, `env.txt`, `startup.txt` and `chat.json`.
  - `dsv41/install.log`, `dsv41/runs/dl_e128t.log`, `dl_watchdog.log`.
- **Secret scan:** `data_tools/secret_scan.py` over all 52 files: 0 hits. The traces are included in the scan.
