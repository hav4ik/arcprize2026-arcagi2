# Commit run v3 (kernel chankhavu/arc2-qwen38-tools-submit, version 3)

Outputs: `kaggle/runs/v3-v3/`. v2 comparison: `kaggle/runs/v2-v2/`.

**Result: COMMIT GATE: PASS.** Status COMPLETE; the run took about 16 min wall (pushed 22:56 UTC, done about 23:12).

| Item | v3 | v2 |
|---|---|---|
| GPU | 4x NVIDIA L4 (23034 MiB, sm8.9), TP=4. No T4 fallback | same |
| Server ready (phases_s) | 548 s (9.1 min; log says "ready in 410s" from launch) | 562 s (9.4 min) |
| Concurrency | 29 (KV 1,402,470 tokens / 48k) | 35 |
| Attempts / answered / errors | 5 / 4 / 0 (stream_errors 0) | 5 / 2 / 0 |
| Score (local, 4 tasks) | 2.00/4 (attempt_1 only is also 2.00/4) | 2.00/4 |
| tool_calls / tool_errors / parse_failures | 26 / 1 / 0 | 21 / 2 / 0 |
| kv_usage_max_pct | 5.3 | 4.5 |
| Prefix-cache hit (overall) | 60.9% (27.6% early, 88-91% once warm) | not logged |
| Preemptions | 0 | 0 |
| Decode throughput | 75 tok/s aggregate (28,059 tokens in 6.2 min); 61-66 tok/s per engine log with 3 reqs | similar (73-78 tok/s) |

Answer sources: submit 2, forced 2, null 1. Train check: 1 pass, 2 fail (the forced ones), 2 not recorded.

Attempt_2 sources (slot2_sources): free alternatives 0, gated attempts 0 (eligible 4, started 0, dropped_time 1 because "no time left"), fallback 2 (input copy, for the two correct tasks), dup 3. The free attempt_2 path was never exercised: `n_alt` was 0 for every pair, so `with_alternatives` = 0. The gated attempt 2 was never started either, because of the 6 min deadline.

submission.json: valid. 240 task ids = test challenges, the number of test items matches per task, every item has attempt_1 and attempt_2 as non-empty rectangular int grids with colors 0-9.

Log scan: no Traceback, MemoryError, OOM or pkill lines (log and vllm_server.log). The v3 self-test `session_mem: True`. The session memory cap (2048 MB) was active.

Red flags and notes:
- The attempt_2 features (free alternatives, gated 2nd attempt) were not exercised by this short deadline run. That means the 2 GB session cap under load and the alt path are untested on Kaggle.
- Score equals v2's 2/4 with the same failing tasks (dd6b8c4b x2 and 58f5dbd5 hit output_budget at about 6.7-7k tokens because of the 6 min forced deadline).
- Concurrency dropped from 35 to 29 by design (KV / 48k).
