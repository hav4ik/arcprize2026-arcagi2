# E059: server restart under load with KV offload (2026-10-27)

Question: when the vLLM engine dies mid-run with KV CPU offload on, does the solver's restart path stay inside Kaggle's
173 GiB host limit? The old server holds Engram (~91 GiB shared) plus the pinned offload pool (16 or 32 GiB); if the new
server allocates its own before the old memory is released, host memory briefly doubles. Also checked: jobs re-queued
without losing tries (S15), throughput recovery, no capacity cliff afterwards.

Design: candidate B exactly (pinned_candb code, 56 workers, gate 0.85 + queue hold), the 90 heaviest tasks of
`evalx2t_s2026`, a 3 h run (stop-new 2.67 h, force 2.83 h, hard 2.92 h), `killer.sh` SIGKILLs this run's own
VLLM::Worker_TP1 (found among the solver's descendants, never by name) 1.5 h after "server up" and samples host memory
every 2 s from 5 min before to 20 min after. Pass: restart peak (anon + shmem) <= ~1.1x steady, jobs re-queued, >= 90%
of pre-kill throughput within ~15 min.

Arms: `launch_b32kill.sh` (32 GiB, Runpod 4xL4 hegn, started 2026-10-27 15:16 UTC); 16 GiB on Vast 4x4090 57983124
(`README_b16_4090.md`, sub-agent).

## Result: 32 GiB on L4 hegn (`b32kill_heavy`): PASS
| time (UTC) | event |
|---|---|
| 16:51:57 | SIGKILL VLLM::Worker_TP1 (pid 114125, a descendant of solver 113067); before: 422-442 tok/s, KV 0.98-0.99, 44-50 running, host restores 3.5-8.6k tok/s |
| 16:52:06 | the solver re-queues every in-flight attempt after "EngineCore encountered an issue" as a server-side error (S15): 63 re-queue lines, 0 attempts ended as errors, one harvested session answer kept |
| 16:52:08 | "server died" detected (11 s after the kill); all VLLM processes gone |
| 16:52:02-16:52:53 | old memory released: shared 123.2 -> 97.7 -> 59.8 -> 9.8 -> 0.2 GiB, anon 17.3 -> 5.1 GiB |
| 16:52:33 | restart 1 starts (the new server allocates only after the old memory is gone) |
| 16:54:53 | server up after 140 s (model files in page cache; on Kaggle expect ~10 min from the mount); KV pool 1,556,850 tokens, unchanged |
| 17:06 / 17:11 / 17:17 | 299 / 459 / 485 tok/s: >= 90% of pre-kill throughput 19 min after the kill (the host pool was lost, so prefixes re-prefill) |

Memory: steady anon+shared before the kill 144.4 GiB (max); peak over the 25-min window 151.7 GiB at 17:10 (anon 28.6 + shared 123.2), ratio 1.05 <= 1.1; min MemAvailable 334.5 GiB. No doubling: the solver stops the old process group (S5) and the new server allocates Engram and the pinned pool only after the old ones are freed.
Cost of one engine death: ~2.5 min down on a pod (~10 min on Kaggle, model reload from the mount) + ~15 min of reduced throughput while caches refill.
Logging gap found: the first metrics window after a restart shows negative rates (server counters reset to 0); fix-list S19 (logging only).

Fix during the run: killer.sh's fallback looked for the solver as `$3` (`python <path>`), but the notebook runs
`python -u <path>`; it now matches the last field, and the L4 run got its `solver.pid` written by hand before the kill.

## OOM kill-order test (local, cgroup v2): PASS
`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python oom_order_test.py`: a stand-in server (3.5 GiB shared + 0.5 GiB anon, oom_score_adj 0) and 4 real `LocalExecutor` sandboxes (oom_score_adj 1000) each allocating 1.2 GiB at once (9.3 GiB > 6 GiB): sandboxes 0-2 OOM-killed and restarted, sandbox 3 held its memory, the server stayed alive, all sessions worked afterwards.
