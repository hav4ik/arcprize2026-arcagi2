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
(`README_b16_4090.md`, sub-agent). Status: running; results to be added here.
Fix during the run: killer.sh's fallback looked for the solver as `$3` (`python <path>`), but the notebook runs
`python -u <path>`; it now matches the last field, and the L4 run got its `solver.pid` written by hand before the kill.
