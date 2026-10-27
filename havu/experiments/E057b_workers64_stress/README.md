# E057b: B+ worker count under heavy KV pressure (56 vs 64 workers, 4x4090)

Status (2026-10-27 12:45 UTC): **done. Verdict: 64 workers PASSES against 56 for B+.**

| | 64 workers | 56 workers |
|---|---|---|
| gen tok/s, server up → force | 774 | 765 (+1.2% for 64) |
| attempts/h | 60.6 | 59.3 |
| preemptions | **0** | **0** |
| server queue | start burst only | start burst only |

Further results:
- **No host-pool cliff in either arm.** In-flight context peaked at **2.62M** (64) and **2.68M** (56) tokens, never
  near 4.5M. No 180-s window after warm-up had `waiting > 0`, so 0 min with host-hit r < 0.5.
- **Load:** KV ran hotter at 64: time above 0.95 was 7.8% vs 5.0% (13.8% vs 8.7% in the last 40 min). The gate absorbed
  it: held 62% vs 53% of samples, 100% vs 95% in the last 40 min. 0 full re-prefills in both arms.
- **Memory:** both arms stayed at 49% of host RAM (AnonPages + Shmem 123.9 / 123.2 GiB).
- **Caveats:** one draw per arm, 80-min deadlines, and 4090 rather than L4 timing (see Caveats).
- **Arm C** (56 workers @ 32 GiB) was not run: the user has not approved going above half the box's RAM, and arm A
  showed no cliff at 16 GiB.

**Question.** B+ is candidate B plus E051 overlay v2 (offload scheduler `94b06f85`), run with 64 workers instead of 56.
Does 64 workers hold up under heavy KV pressure compared with 56?
- The solver's KV gate can only stop admissions, never evict. It holds at `ARC_KV_ADMIT_MAX=0.85`, and with
  `ARC_ADMIT_MAX_WAITING=1` it also holds while the server queues requests.
- With 64 workers all admitted early, growing contexts may saturate the KV pool.
- A second question came from another session: is there a host-pool capacity cliff at 16 GiB? Past about 4.5-4.8M tokens
  of in-flight context, the host-hit ratio collapses to ~0 and throughput drops to the no-offload level. Does
  heavy-first load reach it?

**Verdict rule** (set before the runs). 64 passes if all of these hold; otherwise 56 is recommended for B+:
- its gen tok/s and attempts/h are >= 56's, within noise;
- the server queue is not sustained;
- its preemptions are <= ~2x 56's.

(E057 is the planned q8 all-reduce, hence E057b.)

## Box and setup

- **Box.** Vast 4x RTX 4090, instance **57937248**: Japan, machine 17737, image `vastai/vllm:v0.32.0-cuda-13.0`,
  driver 590.48.01.
  - Direct ssh: `ssh -p 21455 root@106.185.159.136`. The Vast proxy `ssh4.vast.ai:31456` was not used.
  - GPUs: 24,564 MiB each, power limits 450/450/480/450 W, PCIe Gen4 x16. All four sit on one NUMA node: topology NODE,
    with PHB between GPU2 and GPU3.
  - CPU: AMD EPYC 7V13, 128 threads.
- **Memory.**
  - `/proc/meminfo` MemTotal is **251.5 GiB**: half is 125.8 GiB and 55% is 138.3 GiB.
  - The container uses cgroup v1: `memory/memory.limit_in_bytes` is 259,246,784,512 B (**241.4 GiB**). There is no v2
    `memory.max`.
  - `/dev/shm` is 125 GiB, and the disk is a 204 GB overlay.
  - The rule for shared boxes is at most half of host RAM, which is below one B/B+ server (~120-140 GiB). The
    coordinator relayed the user's decision to continue on this box at **<= ~55%**. `run_arm.sh` enforces this with a
    memory guard (below).
- **Image services.** I ran `supervisorctl stop vllm model-ui`. As on earlier 4090 boxes, this left the image's Qwen
  vLLM (pgid 1358: `vllm serve`, EngineCore, 4 workers) holding 5.2 GB per GPU. I stopped it by PGID: SIGTERM, then
  SIGKILL to the group. GPUs went back to 1 MiB.
- **Stack.** E042 `setup_box.sh` → `box_install.sh`: INSTALL OK in 8.3 min.
  - torch 2.13.0+cu130
  - vLLM `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`
  - flashinfer `0.6.18+glm53...`
  - humming 0.1.12
- **Model.** `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` at `9bce98c5`, token from SSH stdin only. See
  "Download incidents" below.
- **On-box chain.** `scripts/setup_chain.sh` follows E056 box 57929406's chain. Its log is `results/setup_chain.log`.
  1. `podprep.sh`: kit `a9706290`, 12-file overlay, toolenv, eval data.
  2. E056 `setup_candb_venv.sh` → `/root/dsv41/venv_candb`: 16/16 sha lines, E051 tests 26/26 OK.
  3. `scripts/setup_candbp_venv.sh` → `/root/dsv41/venv_candbp`.
     - `cp -a` of venv_candb, with 116 `bin/` shebangs rewritten.
     - `install_overlay_v2.sh`: only `offloading/scheduler.py` changes, `dffd7d86` → **`94b06f85`**. Indexer
       **`9f13485b`**; the other 15 candidate-B lines match.
     - venv_candb is unchanged.
     - `vllm` and the v2 scheduler import from the copy.
     - E051 `test_dsv41_kv_offload` + `test_dsv41_kv_offload_v2`: **31/31 OK**.
  4. `/root/kit_bp` = `/root/kit` with `kaggle/dsv41_tools/pinned_candb/` staged at their kit paths: solver
     **77fe80a3**, harness **882c817f**, train_check **8fe36175**, auto_repair `2292369c`, names `571f47eb`.
  5. Workload `evalx2t_s2026` in `/root/data`: challenges `83b249b1398d`, solutions `8ed7202a78eb`, heavy90 ids
     `3c6db22a194e`.
  6. Memory gate at 55%.
  7. One B+ server start (`scripts/verify_candbp_start.sh` → `scripts/serve_candbp.sh`, E056's serve script with the
     55% plan check).

## Download incidents (08:33-10:01 UTC, 88 min instead of ~25)

The E056 script `dl_e128_full.sh` uses `hf download` (huggingface_hub 2.3.0 with hf_xet). It stalled **4 times**
(08:53, 09:15, 09:25 and 09:35): network rx fell to 0.2-0.5 MiB/s with no remote connections, after 6-20 min at
60-90 MiB/s.
- **Lost progress.** Each restart (`kill` of our own download process group by PGID, then a fresh start with the token
  over SSH stdin) started the two Engram shards over: `model-engram-0000{1,2}.safetensors`, 35.9 and 34.3 GB. Xet
  `.incomplete` names carry a per-attempt suffix.
- **Orphans.** The dead attempts' `.incomplete` files filled the disk to 41 GB free. I deleted only those not held open
  by the running `hf` process.
- **Fix.** `results/setup/dl_watchdog.sh` (local) restarts the download after 3 checks with no growth **and** rx below
  5 MiB/s. Xet writes in bursts, so file growth alone gives false stalls. Its final version restarts with plain HTTP
  (`/root/dsv41/dl_e128_http.sh` = `dl_e128_full.sh` + `HF_HUB_DISABLE_XET=1`), which resumes with Range requests.
- **Result.** The last restart (09:39, HTTP) ran at 50-60 MiB/s with no stall. **DL_DONE at 10:00:56:
  131,090,911,045 B, 121 files** (checked by the chain).
- **Chain.** Each restart ended the running chain at step 6 ("process gone without DL_DONE"). `setup_chain.sh` gained
  `FROM=6`, and `scripts/go.sh` resumed it.
- **For the next box:** use `HF_HUB_DISABLE_XET=1` from the start, or the watchdog. The token never touched disk or a
  command line.

## Verify start (B+ server from venv_candbp, 10:02-10:07 UTC)

`verify_candbp_start.sh` → `serve_candbp.sh`, with E056's smoke flags plus 16 GiB offload, util 0.92 and
`--num-gpu-blocks-override 21490`.
- **Ready after 280 s.**
- Offload line once: `KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups [8]`.
- `Overriding num_gpu_blocks=29139 with num_gpu_blocks_override=21490`, so **GPU KV cache size: 1,591,736 tokens**,
  the same as E056's 4090 boxes.
- Indexer skip on in all 6 processes.
- **Tracebacks: 0 at ready and after the chat.** The chat returned `42`, finish `stop`.
- **Host memory:** Shmem 107.0 GiB, server PSS 119.5 GiB at ready; MemAvailable fell 240.2 → 121.5 GiB.
- **Stop:** by PGID; GPUs went back to 1 MiB.
  - The one traceback in `server.log` comes from that stop: the API server's `EngineDeadError` while the whole group
    gets SIGTERM. It is vLLM's usual shutdown noise (E056 box 57929406 saw its EngineCore force-killed the same way).

## Arms (exact settings)

Each arm uses the real solver through `/root/podrun.sh` + `/root/local_run.py`, as the L4 replicas do (E055
`replica_run.sh`, `launch_candBp_x2t.sh`). `scripts/run_arm.sh <name> <workers> [offload bytes]` launches it;
`scripts/stress_chain.sh` runs A then B, one server at a time.

- **Code and venv:**
  - `ARC_KIT=/root/kit_bp`
  - `ARC_VENV=/root/dsv41/venv_candbp`
  - `PATH=/root/dsv41/venv_candbp/bin:/usr/local/cuda-13.0/bin:...`
- **Workload:** `ARC_CHALLENGES` / `ARC_SOLUTIONS` = `evalx2t_s2026`, and `ARC_TASK_IDS` = the 90 heaviest tasks
  (130 pairs, 40 multi-test).
- **Solver settings:** `ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 ARC_CONTEXT_LIMIT=163840 ARC_KV_ADMIT_MAX=0.85
  ARC_KV_ADMIT_MIN_WORKERS=16 ARC_ADMIT_MAX_WAITING=1 ARC_AUTO_REPAIR=1 ARC_MAX_NUM_SEQS=64 ARC_GPU_UTIL=0.92`, plus a
  fresh `ARC_CACHE_SALT` per arm.
- **Deadlines from T0:** `ARC_STOP_NEW_S=4200 ARC_FORCE_AT_S=4500 ARC_HARD_STOP_S=4620 ARC_KILL_AT_S=4800`.
- **Server arguments:** `ARC_SERVER_EXTRA_ARGS` is candBp_x2t's list (Humming, FULL graphs cap 64, retention 512,
  OffloadingConnector at 16 GiB, private, checkpoint, lru) **plus `--num-gpu-blocks-override 21490`**. The script checks
  that it equals that list plus the override.
- **Environment:** the image's credentials and its Qwen `VLLM_*` variables are unset before the launch; Kaggle has none
  of them.
- **Samplers:**
  - `sampler.py`, every 10 s: `/metrics` (all series, plus the TTFT and queue-time buckets), the solver's
    `status.json`, and `/proc/meminfo` plus the cgroup.
  - memdetail, every 30 s: AnonPages, Shmem, MemAvailable, cgroup, GPU MiB, and the server group's
    Pss_Anon+Pss_Shmem. Its cgroup field read only the v2 file, so it is empty in arm A; the v1 fallback went in before
    arm B (file replaced by rename, so the running arm was unaffected). `sampler.py` had the v1 fallback in both arms.
  - `nvidia-smi`, every 30 s.
- **Memory guard:** if the server process group's Pss_Anon+Pss_Shmem exceeds 55% of MemTotal in 2 samples in a row,
  the solver's process group gets SIGTERM. The solver then writes its submission and stops its server.

## Results

- **Order and timing.** The coordinator's update set the order A = 56 first (the smoke-tested Kaggle config), then
  B = 64; both at 16 GiB.
  - Arm A (`e057b_w56`): T0 10:07:13 UTC, server up at +313 s, done at +1.26 h.
  - Arm B (`e057b_w64`): T0 11:23:19 UTC, server up at +283 s, done at +1.26 h.
  - Both runs: 0 restarts, 0 errors, 0 tracebacks during serving, solver exit 0, server group gone, GPUs back to 1 MiB.
- **Windows** (solver clock):
  - **full** = server up → force deadline (4500 s);
  - **last 40** = 2100-4500 s;
  - **open** = server up → admission close (56: 3432 s; 64: 3287 s).
- **Admission close.** The solver's min-request-tokens rule (time to force × median per-stream rate < 15,000 tokens)
  closed admission at +0.95 h (56: 14.0 tok/s per stream, 96 of 180 jobs never started) and +0.91 h (64: 12.3 tok/s,
  91 unstarted).
- **Source.** `scripts/compare_arms.py` produced the table. Full output: `results/compare_w56_w64.{json,md}`.

| metric | 64 full | 56 full | 64 last 40 | 56 last 40 | 64 open | 56 open |
|---|---|---|---|---|---|---|
| **gen tok/s** | **774.4** | **765.0** | 805.8 | 797.0 | 782.8 | 779.5 |
| generated tokens in window | 3.26M | 3.20M | | | | |
| **finished attempts/h** (natural ends) | **60.6** | **59.3** | 73.5 | 73.5 | 59.9 | 51.9 |
| pairs/h (multi-test attempt = its pairs) | 85.4 | 83.4 | 106.5 | 106.5 | 79.1 | 71.6 |
| attempts natural / cut at force | 71 / 17 | 69 / 15 | 49 / 17 | 49 / 15 | 50 / 0 | 45 / 0 |
| attempts ended without an answer | 16 (all cut) | 14 (all cut) | | | | |
| per-stream tok/s, median of natural attempts | 13.5 | 15.0 | 14.8 | 16.4 | 12.3 | 14.0 |
| KV mean / peak | 0.751 / 0.991 | 0.715 / 0.968 | 0.871 | 0.846 | 0.722 | 0.696 |
| time with KV > 0.90 / > 0.95 | 24% / 7.8% | 21% / 5.0% | 41% / 13.8% | 37% / 8.7% | 27% / 10.3% | 27% / 6.7% |
| **preemptions** (total, per h) | **0, 0** | **0, 0** | 0 | 0 | 0 | 0 |
| server running, mean | 46.7 | 43.6 | 38.1 | 37.7 | 54.2 | 49.5 |
| server waiting: mean / max / share of samples > 0 | 2.19 / 60 / 14% | 1.52 / 51 / 12% | 0.08 / 2 / 6% | 0.07 / 2 / 5% | 3.06 / 60 / 19% | 2.03 / 51 / 15% |
| solver attempts running (mean) | 52.2 | 48.3 | 42.3 | 41.8 | 58.2 | 52.5 |
| **gate held** (queued > 0 and running < workers, before stop-new) | 62% | 53% | 100% | 95% | 51% | 41% |
| gate log lines: KV usage / queue holds | 5 / 1 | 4 / 2 | | | | |
| GPU prefix-hit share of queried tokens | 0.729 | 0.754 | 0.699 | 0.695 | 0.601 | 0.666 |
| **host-restored tokens** (`external_prefix_cache_hits_total`) | **14.68M** (3,487 tok/s) | **12.64M** (3,025 tok/s) | 12.53M (5,241/s) | 12.17M (5,093/s) | 13.81M | 11.79M |
| **true uncached share** (queried − GPU hits − host hits) / queried | **3.5%** (511 tok/s) | **3.3%** (468 tok/s) | 1.3% | 1.4% | 5.0% | 4.4% |
| TTFT mean (s); p50 / p90 bucket bound | 7.0; 1.0 / 5.0 | 5.4; 1.0 / 2.5 | 0.94; 0.75 / 2.5 | 0.91; 0.75 / 2.5 | 8.5 | 6.4 |
| queue time mean (s) | 5.9 | 4.4 | 0.25 | 0.24 | 7.4 | 5.3 |
| in-flight context mean / peak (tokens) | 1.97M / 2.62M | 1.88M / 2.68M | 2.29M | 2.26M | 1.91M | 1.86M |
| minutes in-flight > 4.5M | 0 | 0 | 0 | 0 | 0 | 0 |
| host-hit r, warm 180-s windows with waiting > 0 | 0 windows | 0 windows | | | | |
| minutes r < 0.5 sustained ≥ 360 s (that definition) | **0** | **0** | 0 | 0 | 0 | 0 |
| r over all warm windows: median / p10 | 0.83 / 0.10 | 0.60 / 0.12 | 0.91 / 0.29 | 0.92 / 0.26 | | |

Notes on the table:
- **Full re-prefills** (from the traces: turn > 1, prompt ≥ 50k, cached ≤ 640) were **0 of 448** (64) and **0 of 457**
  (56). Uncached prompt tokens after turn 1 were 1.8% and 1.7%.
- **The host-hit ratio r** is Δ`external_prefix_cache_hits` / Δ`external_prefix_cache_queries` over consecutive
  180-s windows (`host_ratio_windows` in the json).
  - **Coordinator's definition (windows with waiting > 0):** the only qualifying windows are the start bursts, with
    r = 0 on cold prompts. After warm-up the server almost never queued, so neither arm shows a cliff signature.
  - **Over all windows:** r was 0.88-0.99 in both arms through the high-KV phase (64: 1903-3163 s; 56: 2293-3553 s).
    64 then held 0.72-0.95 until 3883 s.
  - **Where r was low:** at 10-30 min (≈0.10-0.18) and in the final drain after admission closed (≈0.24-0.29). Early
    on, queries were mostly new first-turn prompts while KV was below 0.8, so little had been offloaded yet. In the
    drain the remaining queries are again mostly new content.
  - Gen tok/s stayed at 780-940 in the early phase. In the drain it fell to 590-740 only because fewer attempts
    were running.
- **In-flight context** comes from the traces. Each attempt's turns are spread evenly over [end − wall_s, end], with
  linear interpolation of per-turn `total_tokens`. Cut streams' `BIG` (10^9) completion counts are removed.
  - 64 workers peaked at +2500 s and 56 workers at +2850 s.
  - The peaks are 1.6-1.7x the 1.59M-token GPU pool, so ~1M tokens lived only in the host pool or were idle between
    turns.
  - The heavy-first queue never approached the 4.5-4.8M cliff region within these runs.
- **Memory** (cgroup v1, `memory.limit_in_bytes` 259,246,784,512 = 241.4 GiB; MemTotal 251.5 GiB):

  | | 64 workers | 56 workers |
  |---|---|---|
  | peak AnonPages | 16.9 GiB | 16.3 GiB |
  | peak Shmem (Engram ~91 + 16 GiB pinned pool) | 107.0 GiB | 107.0 GiB |
  | sum | **123.9 GiB (49.3%)** | **123.2 GiB (49.0%)** |
  | server group Pss_Anon+Pss_Shmem, max | 121.1 GiB | 120.9 GiB |
  | MemAvailable, min | 117.4 GiB | 117.5 GiB |

  - The 55% guard (138.3 GiB) never fired.
  - The cgroup's usage sat at ~240 GiB only through page cache (the model read at staging), which is reclaimable.
- **GPU:** memory peak 22.44-22.50 GB per GPU in both arms. Mean util 85-87% (64) vs 82% (56); mean power 203-245 W.
- **Accuracy** (1 draw each, report only):
  - Local score: 64 = **86/130 pairs**, 56 = **80/130**.
  - Pairs answered by finished attempts: 106 vs 98, of which 81.1% vs 81.6% are correct.
  - The difference is coverage (64 started 89 jobs vs 84) within draw noise. Do not over-interpret.

### Verdict (rule set before the runs)

| condition | 64 vs 56 | pass? |
|---|---|---|
| gen tok/s ≥ 56's, within noise | 774 vs 765 full (+1.2%); 806 vs 797 last 40 min; 783 vs 780 open | yes |
| attempts/h ≥ 56's, within noise | 60.6 vs 59.3 full; 73.5 vs 73.5 last 40 min; 59.9 vs 51.9 open | yes |
| no sustained server queue | waiting > 0 only in the start burst (max 60 vs 51); last 40 min mean 0.08 vs 0.07 | yes |
| preemptions ≤ ~2x 56's | 0 vs 0 | yes |

**64 workers passes: keep 64 for B+.** The gate does its job at 64: it holds admissions at KV ≥ 0.85, and running
attempts drift from 63-64 down to ~40 as contexts grow, much as 56's drift from 55 to ~39. The price is a hotter pool:
- time above KV 0.95 is 1.6x as long;
- the per-stream rate is ~10% lower, so each attempt runs slower;
- TTFT in the start burst is higher (7.0 vs 5.4 s mean).

None of this turns into preemptions, queueing or re-prefills on this workload.

## Caveats

- **One draw per arm.** The stack is not run-to-run deterministic. Differences of a few percent in tok/s or
  attempts/h, and of 6 pairs in the score, are inside the noise.
- **4090, not L4.**
  - 4090 prefill and decode are much faster than Kaggle's 4xL4. The same 80 min covers more model work per attempt, and
    any re-prefill or preemption costs less wall time here. Contention costs are therefore understated, although none
    occurred.
  - The KV pool is pinned to 1,591,736 tokens (21,490 blocks at util 0.92), +2.4% over Kaggle's 1,554,183.
- **Short deadlines.**
  - Stop-new 70 min and force 75 min, against Kaggle's ~11 h.
  - Admission closed at ~55-57 min, so each arm covers roughly the first wave of attempts on the 90 heaviest tasks.
  - Attempts/h counts natural ends inside the window and is dominated by when the first wave finishes (mean attempt
    wall time ~33 min).
  - A 12 h run reaches later phases (attempt 1, mixed context ages) that these arms do not.
- **The host-cliff question is answered for this load only.**
  - In-flight context peaked at ~2.6-2.7M tokens. To reach 4.5M at 64 workers, the average in-flight context would
    need ~70k tokens per attempt.
  - Neither arm got there because the gate shed concurrency as contexts grew.
  - A harder or longer mix could still get there. That is arm C's territory, not run.
- **Box.**
  - Host RAM was shared at 55% by the user's decision, relayed by the coordinator; both arms used 49%.
  - Tool sandboxes ran on a 128-thread EPYC, which is not Kaggle's CPU.

## How to reproduce

```bash
# on the box, after setup (scripts/setup_chain.sh; FROM=6 resumes after a download restart)
setsid -f bash /root/e057b/scripts/stress_chain.sh > /root/e057b/runs/stress_chain.driver.out 2>&1 < /dev/null
# or one arm:  bash /root/e057b/scripts/run_arm.sh <name> <workers> [offload bytes]
# locally, after copying /root/runs/<name> back (secret-scanned):
.venv/bin/python experiments/E057b_workers64_stress/scripts/compare_arms.py \
  experiments/E057b_workers64_stress/results/runs/e057b_w56 experiments/E057b_workers64_stress/results/runs/e057b_w64 \
  --out results/compare_w56_w64.json --md results/compare_w56_w64.md
```

## Files

- **`scripts/`:**
  - `setup_chain.sh`: the on-box bring-up chain (steps 1-8; `FROM=6`).
  - `setup_candbp_venv.sh`: venv_candbp, overlay v2, 31 tests.
  - `serve_candbp.sh`: E056's `serve_candb.sh` with the 55% plan check and the E057b run dir.
  - `verify_candbp_start.sh`.
  - `go.sh`: chain step 6+, then the stress chain.
  - `run_arm.sh`: one arm, samplers and memory guard.
  - `stress_chain.sh`: A = 56, then B = 64.
  - `sampler.py`: 10-s `/metrics`, status and memory.
  - `compare_arms.py`: all metrics in this README.
- **`results/compare_w56_w64.{json,md}`:** per-arm metrics for the three windows, the 180-s host-ratio windows and the
  10-s in-flight context series.
- **`results/runs/e057b_w{56,64}/`:** each arm's run dir (no scratch).
  - `run.log`, `env.txt`, `arm_t0`, `metrics10.jsonl`, `memdetail.log`, `gpu.csv`.
  - `out/dsv41_logs/`: `solver.log`, `attempts.jsonl`, `metrics.jsonl`, `server.log`, `server_cmd.json`,
    `status.json`, `stage.log`.
  - `out/dsv41_traces.tar.gz` and `out/submission.json`.
- **`results/setup/`:**
  - `e057b/runs/`: `setup_chain.log`, `setup_candbp_venv.log`, `e051_tests_candbp.log`, `podprep.out`,
    `candb_{before,after}.sha256`, `verify_candbp.progress`, `verify_candbp/{server.log, env.txt, startup.txt,
    chat.json}`, `stress_chain.log`, `e057b_w{56,64}.progress`.
  - `e056/runs/`: `setup_candb_venv.log`, `notebook_install.log`, `e051_cpu_tests.log`.
  - `dsv41/install.log` and `dsv41/runs/dl_e128t.log`.
  - `dl_watchdog.sh`, the local download watchdog, and `dl_e128_http.sh`.
- **Secret scan:** `data_tools/secret_scan.py` found 58 files, 0 hits. The image's own credential variables were also
  grepped for: 0, including inside the traces.
