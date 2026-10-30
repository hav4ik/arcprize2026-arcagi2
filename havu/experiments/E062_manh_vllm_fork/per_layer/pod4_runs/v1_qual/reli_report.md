# Reliability report: raw/runs/v1_qual/mon.jsonl
VERDICT: N/A: Kaggle-eq RAM: 131.75 GiB now, too little post-warm-up data for a slope; worst cap - peak: 2.21 GiB now, too little post-warm-up data for a slope
samples 876 (529 fast), 2026-10-30T09:21:10Z .. 2026-10-30T15:51:17Z; clock: Kaggle h (solver t0 2026-10-30T09:21:27Z)
host RAM source: cgroup_v1; /proc/meminfo is host-wide here (+79.4 GiB not ours at the last sample)

## Snapshots (every 30 min)
| Kaggle h | UTC | Kaggle-eq | meminfo form | cg anon | cg shmem | /dev/shm | sandboxes n / GiB | engines | worst peak | worst reserved | cap-peak | cap-reserved | retries | OOMs | smi free min | gen tok/s | KV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.00 | 09:21:10 | 7.08 | 89.09 | 0.08 | 0.00 | 0.00 | 0 / 0.00 | 0 | - | - | - | - | - | - | 22.04 | - | - |
| 0.01 | 09:22:10 | 15.64 | 97.74 | 0.21 | 8.43 | 8.43 | 0 / 0.00 | 0 | - | - | - | - | - | - | 22.04 | - | - |
| 0.51 | 09:52:10 | 133.34 | 212.40 | 119.66 | 6.68 | 0.36 | 64 / 2.97 | 1 | 18.97 | 19.39 | 2.18 | 1.76 | 0 | 0 | 2.01 | 727.80 | 0.65 |
| 1.01 | 10:22:10 | 133.57 | 210.46 | 119.89 | 6.68 | 0.36 | 55 / 2.91 | 1 | 19.02 | 19.54 | 2.13 | 1.61 | 0 | 0 | 1.84 | 679.60 | 0.89 |
| 1.51 | 10:52:10 | 133.29 | 210.30 | 119.61 | 6.68 | 0.36 | 34 / 1.99 | 1 | 19.13 | 19.66 | 2.02 | 1.49 | 0 | 0 | 1.74 | 536.00 | 0.90 |
| 2.01 | 11:22:10 | 133.50 | 219.97 | 119.81 | 6.69 | 0.36 | 37 / 2.06 | 1 | 19.28 | 19.80 | 1.87 | 1.35 | 0 | 0 | 1.60 | 505.70 | 0.88 |
| 2.51 | 11:52:10 | 133.69 | 220.74 | 120.00 | 6.69 | 0.36 | 42 / 2.30 | 1 | 19.28 | 19.80 | 1.87 | 1.35 | 0 | 0 | 1.60 | 511.40 | 0.89 |
| 3.00 | 12:21:30 | 133.55 | 221.14 | 119.86 | 6.69 | 0.36 | 33 / 1.95 | 1 | 19.31 | 19.84 | 1.84 | 1.31 | 0 | 0 | 1.56 | 512.70 | 0.89 |
| 3.50 | 12:51:32 | 132.33 | 210.68 | 118.65 | 6.68 | 0.36 | 64 / 3.39 | 1 | 18.98 | 19.39 | 2.17 | 1.76 | 0 | 0 | 2.01 | 500.40 | 0.74 |
| 4.00 | 13:21:32 | 132.04 | 209.88 | 118.36 | 6.68 | 0.36 | 48 / 2.70 | 1 | 19.06 | 19.58 | 2.09 | 1.57 | 0 | 0 | 1.82 | 442.00 | 0.90 |
| 4.50 | 13:51:32 | 131.86 | 209.91 | 118.18 | 6.68 | 0.36 | 39 / 2.26 | 1 | 19.10 | 19.62 | 2.05 | 1.53 | 0 | 0 | 1.78 | 399.30 | 0.88 |
| 5.00 | 14:21:32 | 131.58 | 214.72 | 117.89 | 6.69 | 0.36 | 26 / 1.53 | 1 | 19.18 | 19.70 | 1.97 | 1.45 | 0 | 0 | 1.70 | 373.80 | 0.87 |
| 5.50 | 14:51:32 | 131.96 | 215.77 | 118.27 | 6.69 | 0.36 | 43 / 2.21 | 1 | 19.26 | 19.80 | 1.89 | 1.35 | 0 | 0 | 1.60 | 398.50 | 0.89 |
| 6.00 | 15:21:27 | 131.73 | 231.45 | 118.04 | 6.69 | 0.36 | 33 / 1.70 | 1 | 19.34 | 19.86 | 1.81 | 1.29 | 0 | 0 | 1.54 | 398.50 | 0.90 |
| 6.50 | 15:51:17 | 131.75 | 211.15 | 118.07 | 6.68 | 0.36 | 64 / 2.36 | 1 | 18.94 | 19.39 | 2.21 | 1.76 | 0 | 0 | 2.01 | 0.00 | 0.85 |

## Per-hour dynamics (change within each hour, per hour; '-' = under 15 min of data)
| Kaggle h | Kaggle-eq end | /h | worst peak end | /h | worst reserved end | /h | cap-peak end | retries | TP RSS /h | sandboxes /h | gen tok/s | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1-0 | 7.08 | - | - | - | - | - | - | - | - | - | - |  |
| 0-1 | 133.60 | 129.28 | 19.02 | - | 19.54 | - | 2.13 | 0 | 132.04 | 3.07 | 643 | server restart |
| 1-2 | 133.56 | -0.01 | 19.28 | 0.26 | 19.80 | 0.26 | 1.87 | 0 | 0.03 | -0.73 | 547 |  |
| 2-3 | 133.55 | 0.05 | 19.31 | 0.03 | 19.84 | 0.04 | 1.84 | 0 | 0.01 | -0.11 | 520 |  |
| 3-4 | 132.09 | 1.89 | 19.06 | 0.42 | 19.58 | 0.68 | 2.09 | 0 | 0.09 | 2.99 | -3227 | server restart |
| 4-5 | 131.58 | -0.47 | 19.18 | 0.12 | 19.70 | 0.12 | 1.97 | 0 | 0.04 | -1.19 | 403 |  |
| 5-6 | 131.73 | 0.15 | 19.34 | 0.16 | 19.86 | 0.16 | 1.81 | 0 | 0.01 | 0.17 | 393 |  |
| 6-7 | 131.75 | - | 18.94 | - | 19.39 | - | 2.21 | 0 | - | - | 108 | server restart |

## Trends (slope over the last 2.0 h of the current server, 20 min warm-up skipped; 0 points)
| metric | now | slope /h | limit | time to limit | verdict |
|---|---|---|---|---|---|
| Kaggle-eq RAM | 131.75 | - | 173.00 | - | N/A: Kaggle-eq RAM: 131.75 GiB now, too little post-warm-up data for a slope |
| worst cap - peak | 2.21 | - | 0.26 | - | N/A: worst cap - peak: 2.21 GiB now, too little post-warm-up data for a slope |
| worst cap - reserved | 1.76 | - | 0.00 | - | N/A: worst cap - reserved: 1.76 GiB now, too little post-warm-up data for a slope |
| worst peak allocated | 18.94 | - | - | - | - |
| worst reserved | 19.39 | - | - | - | - |
| TP workers RSS | 125.61 | - | - | - | - |
| EngineCore RSS | 1.32 | - | - | - | - |
| sandboxes RSS | 2.36 | - | - | - | - |
| /dev/shm used | 0.36 | - | - | - | - |
| nvidia-smi free min (board; info only) | 2.01 | - | - | - | - |

## Events
- 09:23:10 (Kaggle h 0.03) server instance /root/kwork_v1_qual/dsv41_logs/server/server.log#0: KV pool 1617253 tokens, cap - reserved at start {'TP0': 2.26, 'TP1': 2.24, 'TP2': 2.23, 'TP3': 2.21}
- 12:21:31 (Kaggle h 3.00) fault kill-tp on VLLM::Worker_TP3 pid 19859
- 12:21:45 (Kaggle h 3.01) server died (fault injection)
- 12:24:22 (Kaggle h 3.05) server instance /root/kwork_v1_qual/dsv41_logs/r1/server/server.log#0: KV pool 1617253 tokens, cap - reserved at start {'TP0': 2.26, 'TP1': 2.24, 'TP2': 2.23, 'TP3': 2.21}
- 15:21:32 (Kaggle h 6.00) fault sigstop-engine on VLLM::EngineCore pid 35013
- 15:47:32 (Kaggle h 6.43) server died (fault injection)
- 15:50:12 (Kaggle h 6.48) server instance /root/kwork_v1_qual/dsv41_logs/r2/server/server.log#0: KV pool 1617253 tokens, cap - reserved at start {'TP0': 2.26, 'TP1': 2.24, 'TP2': 2.23, 'TP3': 2.21}

## Fault injections
fault kill-tp at 2026-10-30T12:21:31Z (3.00 h) on VLLM::Worker_TP3 pid 19859: WARN
         +4s  supervisor state up (start 0)
        +11s  solver: [12:21:43 +3.01h] server died (code 1); tail: pid=18722)[0;0m [33mWARNING[0m [90m10-30 12:21:42[0m [90m[utils.py:653][0m [shutdown] Process manager: forc
        +14s  no pre-fault vLLM process on any GPU
        +14s  supervisor state restarting (start 1)
        +21s  all pre-fault vLLM processes gone
        +26s  first new vLLM process
       +176s  supervisor state up (start 1)
       +176s  solver: [12:24:28 +3.05h] server up after 165s (external, start 1); GPU KV cache size: 1,617,253 tokens; Model loading took 14.33 GiB
  PASS  finalized_errors 0 attempt(s) finalized as error
  PASS  server_died      solver saw the death at +11s
  PASS  requeued         83 re-queue line(s) after the death
  PASS  restart_time     server up 176 s after the fault (limit 900 s)
  PASS  tries_spent      0 retry line(s) (each spends one of the job's tries)
  PASS  one_engine       max 1 EngineCore, no old/new overlap; Kaggle-eq peak 132.38 GiB vs 133.55 before (margin 10)
  PASS  ram_limit        Kaggle-eq peak 132.4 GiB (limit 173.0)
  WARN  jobs             60 job(s) with an error record since the fault: 27 finished normally later, 33 pending (lost if the run has ended)
fault sigstop-engine at 2026-10-30T15:21:32Z (6.00 h) on VLLM::EngineCore pid 35013: WARN
         +0s  supervisor state up (start 1)
      +1511s  solver: [15:46:44 +6.43h] server died (code 1); tail: re_client.py:809][0m [shutdown] MPClient: complete | [0;36m(APIServer pid=34594)[0;0m [32mINFO[0m [90m10-30 
      +1560s  all pre-fault vLLM processes gone
      +1560s  no pre-fault vLLM process on any GPU
      +1560s  first new vLLM process
      +1560s  supervisor state restarting (start 2)
      +1722s  solver: [15:50:15 +6.48h] server up after 210s (external, start 2); GPU KV cache size: 1,617,253 tokens; Model loading took 14.33 GiB
      +1725s  supervisor state up (start 2)
  PASS  finalized_errors 0 attempt(s) finalized as error
  PASS  server_died      solver saw the death at +1511s
  PASS  requeued         16 re-queue line(s) after the death
  PASS  restart_time     server up 1722 s after the fault (limit 3000 s)
  WARN  tries_spent      33 retry line(s) (each spends one of the job's tries)
  PASS  one_engine       max 1 EngineCore, no old/new overlap; Kaggle-eq peak 131.75 GiB vs 131.73 before (margin 10)
  PASS  ram_limit        Kaggle-eq peak 131.8 GiB (limit 173.0)
  WARN  jobs             33 job(s) with an error record since the fault: 0 finished normally later, 33 pending (lost if the run has ended)

## Attempts
attempt records 332, per-try error records 132 (re-queued jobs), finalized errors 0; solver: died 2, re-queues 99, retries 33, timeouts 33
jobs with an error record: 60; finished normally later: 27; pending or lost: 33

VERDICT: N/A: Kaggle-eq RAM: 131.75 GiB now, too little post-warm-up data for a slope; worst cap - peak: 2.21 GiB now, too little post-warm-up data for a slope
