# Fidelity record: e062b_crash

Launched 2026-10-28T23:11:29Z on pod 213.173.105.17:19843 by `experiments/E055_kaggle_replica/toolkit/replica_run.sh`.

## Corrections applied
- **Workload:** `evalx2t_s2026` (experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_challenges.json, sha256 83b249b1398d; solutions sha256 8ed7202a78eb).
- **Deadlines shifted 0 s earlier** (pod clock): stop-new 0.67 h, force 0.73 h, hard stop 0.77 h, kill switch 0.80 h; on Kaggle's clock 0.67 h / 0.73 h / 0.77 h / 0.80 h. The solver's clock starts at the notebook's first cell (ARC_T0 in local_run.py cell 0); on Kaggle the offline install (256 s), the toolchain check (17 s) and the FUSE model load (590 s vs ~285 s on a pod) all come after it, so Kaggle's first attempt starts ~600 s later than a pod's (Kaggle bench V1: server up at +0.24 h; pods +0.06-0.10 h).
- **KV pool:** ARC_GPU_UTIL=0.938 (pods: 1,556,850 tokens at 0.938 vs Kaggle's 1,554,183 at 0.945).
- **Memory log** with the cgroup's memory.current (Kaggle's limit: 173 GiB) and per-GPU power / SM clock every 60 s (`mem.log`).
- **Power cap:** not requested (pods run at their 72 W limit; Kaggle's L4s at 50 W).

## Settings
- `ARC_ADMIT_MAX_WAITING=1`
- `ARC_AUTO_REPAIR=1`
- `ARC_CACHE_SALT=e062b_crash_231109`
- `ARC_CHALLENGES=/root/data/evalx2t_s2026_challenges.json`
- `ARC_CONTEXT_LIMIT=163840`
- `ARC_FORCE_AT_S=2640`
- `ARC_GPU_UTIL=0.938`
- `ARC_HARD_STOP_S=2760`
- `ARC_KILL_AT_S=2880`
- `ARC_KIT=/root/kit_b`
- `ARC_KV_ADMIT_MAX=0.85`
- `ARC_KV_ADMIT_MIN_WORKERS=16`
- `ARC_MAX_NUM_SEQS=96`
- `ARC_MULTI_BUDGET=0.5`
- `ARC_MULTI_TEST=1`
- `ARC_SERVER_ENV_JSON={"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64"}`
- `ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--decode-context-parallel-size", "4", "--cp-kv-cache-interleave-size", "64", "--prefix-cache-retention-interval", "512", "--cpu-offload-gb", "1", "--cpu-offload-params", "embed_tokens", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 96}", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"eviction_policy\":\"lru\"}}"]`
- `ARC_SOLUTIONS=/root/data/evalx2t_s2026_solutions.json`
- `ARC_STOP_NEW_S=2400`
- `ARC_VENV=/root/manh/venv`
- `ARC_WORKERS=80`
- `PATH=/root/manh/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin`

Pod pre-flight: GPUs 0, NVIDIA L4, 0 MiB, 72.00 W, 72.00 W, 40.00 W, 2040 MHz; 1, NVIDIA L4, 0 MiB, 72.00 W, 72.00 W, 40.00 W, 2040 MHz; 2, NVIDIA L4, 0 MiB, 72.00 W, 72.00 W, 40.00 W, 2040 MHz; 3, NVIDIA L4, 0 MiB, 72.00 W, 72.00 W, 40.00 W, 2040 MHz; kit 77fe80a3 /root/kit_b/solver_dsv41.py; 882c817f /root/kit_b/repo/evals/run_deepseek_tools.py; 8fe36175 /root/kit_b/repo/evals/train_check.py; cgroup_max: 285999996928; nproc: 48; mem_total_gib: 503.

## Remaining known gaps (corrected or bounded by replica_report.py)
1. **Throughput: pods are faster than Kaggle's 50 W L4s.** Measured on the twin pair Kaggle bench V1 / pod `mt36_b05_gate60` (V6 settings): effective prefill 821 vs 1,187 tok/s (k = 1.45; notes section 7: 1.26-1.45 from saturated rates), pure-decode per-stream rate at 34-36 running 9.60 vs 10.20 tok/s (d = 0.94), generated tokens 1 / 1.5 / 2 h after server up 0.89 / 0.89 / 0.91 x the pod (the stretch below predicts 0.88 / 0.88 / 0.89; `calibrate_twin.py`, in-sample). The report's Kaggle-clock section stretches the pod timeline by (1 + (k - 1) s) / d per 10-s interval (s = prefill share of server time from server.log) and drops attempts that would end after Kaggle's force deadline: central (1.45, 0.94), optimistic (1.3, 1.0), pessimistic (1.6, 0.94). It assumes the same sequence of server states, only slower; in thrash (s up to ~0.75) the gap widens to ~x0.7.
2. **Workload:** eval tasks (public, ~2,607 cells per task, 41% multi-input) stand in for the private 240 tasks / ~335 pairs; a heavier or harder private set shifts KV pressure and accuracy together. Transformed copies are new prompts for the model but the same puzzles: outcomes of a task and its copy are correlated, so the score's spread is that of ~120 independent tasks (the report's cluster bootstrap).
3. **Draw noise:** the stack is not run-to-run deterministic; one replica is one draw (sd ~4 task-weighted points on 120 clusters). Compare candidates on the same workload and seed, paired by task (`replica_report.py --baseline`).
4. **Restarts:** a pod reloads the model from local disk (~110-175 s); Kaggle reloads from the FUSE page cache (untested). The Kaggle-clock stretch does not lengthen outages.
5. **Host:** CPU count and speed differ (tool execution is ~0.5% of attempt time on pods); the pod's `free -g` shows the whole host, so use cgroup / RSS for the 173 GiB limit.
6. **Before T0:** Kaggle's container start and input mounting before the first cell (~6-9 s visible in the kernel log; the rest unknown) count against 12 h but not against the solver's deadlines.
