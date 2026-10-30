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

## Results
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
