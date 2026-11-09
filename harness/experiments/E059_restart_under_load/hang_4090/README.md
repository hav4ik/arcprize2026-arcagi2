# E059 hang rehearsal: B32.1's failure handling on Vast 4x4090 57983124 (2026-10-27)

**Question.** vLLM does not detect a wedged engine (the API server keeps answering `/health` and `/metrics` while nothing
is generated; `notes/2026-10-27_vllm_hang_protection.md`). Does B32.1's progress watchdog (S20) turn such a hang into a
restart, and do the re-launch hold (S21) and the post-restart ramp (S22) behave?

**Setup** (`run_hang.sh`, `../freezer.sh`): the exact B32.1 solver (dee879ce) through the notebook pipeline
(`podrun.sh` + `local_run.py`), with the B32.1 notebook's smoke settings: the 24 smoke-stress tasks (31 pairs, 48
attempts), GPU KV pool cut to 4,500 blocks (333,309 tokens), 32 GiB KV offload, 56 workers, gate 0.85 + queue hold,
`ARC_STALL_RESTART_S=60` (the smoke's shortened watchdog; 600 in the competition rerun), `ARC_CONN_HOLD=1`,
`ARC_RESTART_RAMP=16,8`, `ARC_RESTART_MIN_LEFT_S=300`, 34-min deadlines. `freezer.sh` sends SIGSTOP to this run's own
`VLLM::EngineCore` (found among the solver's descendants) 20 s after "server up", as the notebook's cell 2c does.

**Result: PASS.**

| step | time (UTC) | |
|---|---|---|
| server up (first start, 225 s) | 19:48:42 | |
| engine frozen (SIGSTOP EngineCore) | 19:49:08 | 48 attempts in flight |
| watchdog: "server stalled: no generated tokens for 60 s with 48 attempts running (counter flat)" | 19:50:13 | +65 s |
| old server stopped (SIGTERM to the group, SIGKILL after 30 s: a stopped process cannot exit on SIGTERM), restart 1 | 19:50:43 | +95 s |
| server up again (110 s start) | 19:52:33 | outage 3 min 25 s |
| all 48 in-flight attempts re-queued as server-side errors, each on its first connection retry; 0 second retries, 0 job tries spent | 19:52:33 | |
| ramp: 16 attempts at once, 32 a minute later, then the queue ran out | 19:52:33-19:55 | |
| all 48 attempts finished; local score 31/31 pairs; solver exit 0; submission valid | 20:07:22 | |
| after the run: 2 server groups seen, 0 alive; GPUs 1 MiB; no VLLM processes | 20:08:11 | |

Container memory (cgroup anon + shmem, 2-s samples): steady 136.0 GiB before the freeze, minimum 1.5 GiB between the old
and new server, restart peak 134.1 GiB (0.99x steady), 135.3 GiB after. The old server's memory was gone before the new
one allocated, as in the E059 kill tests.

**Notes.**
- The solver's main loop blocks inside the restart (stop + start), so the lost attempts are processed only once the new
  server is up; the S21 race (re-launch onto a dying server) cannot occur on this path. S21 matters when the server
  dies on its own and the solver notices late (the B16 run). It is unit-tested.
- With the competition setting (600 s), a hang costs ~10 min of detection + ~0.5 min stop + the restart (~2-5 min on
  Kaggle) plus the in-flight attempts' generated tokens.
- Not covered here: the same test on Kaggle (cell 2c of the B32.1 notebook does it in a commit run; not run, to save
  quota) and a hang late in a long run under real KV pressure.

Files: `run_hang.sh`, `../freezer.sh`, `results/e059_hang/` (solver, server and restart logs, 2-s memory, 10-s metrics,
traces, submission; secret-scanned).
