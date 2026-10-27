# Kaggle submission testing protocol (minimum, ARC Prize 2026)

Every Kaggle submission follows this protocol. It is the minimum; add tests for whatever the candidate changes. A
submission is a 12-hour run we cannot watch, on hardware and inputs we only partly control, so each stage removes one
way it can fail before we spend a daily slot. Worked example: candidate B32, `experiments/E056_candidate_b/TESTING_B32.md`.

Roles: whoever prepares the candidate runs stages 0-5 and writes the test record; **every Kaggle run (stage 6) and
every submission needs the team lead's explicit OK** (GPU quota is scarce: a 4×L4 session bills 2× against the weekly
quota).

## Stage 0. Freeze the candidate
- Copy the exact solver, harness and helper files into a pinned folder (e.g. `kaggle/dsv41_tools/pinned_<name>/`) and
  record their sha256. Build the notebook only from pinned files, never from the live `evals/` copies (other work
  changes them).
- Build the notebook with `kaggle/dsv41_tools/build_frozen_mt.py`. It hash-checks every file in the notebook, checks
  the settings, and refuses pinned files whose imports the Kaggle kit cannot satisfy.
- Record: notebook slug and version, every file hash, every setting (workers, gate, budgets, server flags, KV pool),
  every input (competition, datasets, kernels, model version).
- From here on, any change to code, settings or inputs restarts the protocol for the affected stages.

## Stage 1. Unit and packaging tests (CPU, minutes)
- `kaggle/dsv41_tools/tests` and `evals/tests` pass.
- Each notebook cell runs in its commit, competition-rerun and fallback paths (tests in `test_notebook_cells.py`).
- New solver/harness behaviour has a regression test that fails on the old code.

## Stage 2. Component tests on a dev box (4×4090 or similar), for every serving change
- A change to model outputs or KV handling: bit-identity of logits or byte-level KV parity against the old code,
  judged against the stack's own run-to-run noise (the stack is not deterministic end to end).
- A change meant to speed things up: an A/B on the same box, same workload, one server at a time.
- A change to accuracy-relevant behaviour (prompts, budgets, answer handling): a 120-task eval A/B, one draw per arm at
  least, read against the ±3-point noise of one draw.

## Stage 3. Local Kaggle replica on 4×L4 (mandatory)
The core test. It runs the frozen notebook code on Kaggle-like hardware with a hidden-set-sized workload.

**Setup**
- A 4×L4 box with ≥ 256 GB host RAM (L4s run at 72 W, as on Kaggle). Setup: `kaggle/dsv41_tools/REPLICATE.md` section 4.
- KV pool pinned to Kaggle's (GPU util 0.938 on Runpod pods = 0.945 on Kaggle; check the "GPU KV cache size" line).
- Workload: 240 tasks / ~344 pairs, the size of the hidden set: 120 public eval tasks ×2 with transformed copies
  (`experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026`), so prompts do not share prefix cache.
- The notebook's own cells via `local_run.py` + `podrun.sh`, launched with `toolkit/replica_run.sh` (Kaggle's deadlines
  moved 600 s earlier for Kaggle's slower startup; memory logging on).

**Duration**
- **Minimum 6 hours of continuous running.** By then the queue has reached heavy tasks, the KV pool has been under
  pressure for hours, and memory growth is visible. A 6-hour run can pass or fail the candidate on stability, but its
  score is partial.
- **For a score estimate, run to the Kaggle-clock force deadline (pod hour ~9.7-10) or the full ~11-12 hours.** The
  score depends on how many tasks finish before Kaggle's deadline, which only the end of the run shows.

**Pass criteria** (from `toolkit/replica_report.py` and the polls)
- No crash, no server restart, no `Traceback` in the solver log; the final submission is written.
- Throughput holds: no sustained collapse (a thrash window = GPU prefix hit < 0.85 with > 500 recomputed tok/s);
  report tok/s per hour.
- KV pressure handled: the gate holds, no sustained server queue, preemptions absorbed (report counts per hour).
- Host memory: server peak (PSS anon + shared) + ~20 GiB margin below Kaggle's 173 GiB; growth flattening; GPU memory flat.
- Coverage: first attempts for all tasks before the Kaggle-clock force deadline, or a stated shortfall.
- Accuracy of answered pairs in line with the harness's offline numbers.

**Record**: raw score, Kaggle-clock score, coverage (first/second attempts), accuracy of answered pairs, tok/s,
output tokens, KV mean/max, preemptions, recomputed vs restored tokens, memory peak; archive the run (attempts, traces,
metrics) in `experiments/E055_kaggle_replica/runs/` and, for shared data, a private HF dataset.

## Stage 4. Targeted stress tests (as relevant to the candidate)
- Heaviest-tasks-first run with the real solver (e.g. the 90 heaviest tasks, ~80 min): the late-run state in an hour.
- Late-phase replay at fixed concurrency (E051 `tool_replay_kv.py`) for serving changes: where does throughput collapse?
- Worker count or gate changes: compare against the current setting on the same heavy workload.

## Stage 5. Failure paths
- Server death under load: kill one worker of our own server mid-run (never by name; `experiments/E059_restart_under_load/killer.sh`)
  and confirm the restart stays under Kaggle's memory limit, jobs are re-queued, throughput recovers.
- Tool sandbox: per-session memory/CPU/file limits hold and host memory is returned (`cells/cell2b_sandbox_check.py`).
- Deadlines: the force cut and the final write run (covered by the replica and the smoke).

## Stage 6. Kaggle smoke of the exact notebook version to submit (team lead's OK)
Build with `--smoke-minutes N --smoke-stress --smoke-integrity <model manifest>`; N = 30-40. The commit run is the smoke;
the competition rerun of the same version is the submission, with the smoke settings inactive.
- **Input integrity:** every model file by size and sha256 against the pre-upload manifest; the wheel set; the kit
  (sha-checked by cell 0); the competition files parse. Teams have hit corrupted datastores; this catches them.
- **Model load and server start** with the exact flags; the KV pool line equals the replica's; startup lines expected
  for the candidate (e.g. the KV offload line).
- **Tool sandbox at full load:** as many sessions as workers; cap and timeout checks.
- **Inference under KV pressure:** a small GPU KV pool and ~24 tasks so evictions, host restores, preemption and the
  gate all run in minutes.
- **Force cut, final write, valid submission.**
- **Report:** host memory peaks (anon / shared / file) against 173 GiB, GPU memory, power and clocks, free disk, sandbox
  peak, and the coverage line.
Pass: the commit run completes without "CHECK FAILED", "SANDBOX CHECK FAILED" or "INPUT INTEGRITY FAILED"; the coverage
line shows the paths ran; memory has margin. Budget: ≤ 45 min wall (≤ 1.5 h quota). Watch the run; ask before cancelling.

## Stage 7. Score estimate and sign-off
- Estimate from the replica's Kaggle-clock score, calibrated against earlier submissions' replica-vs-leaderboard
  gaps. Quote a range (one replica draw has sd ~3-4 points; the leaderboard draw another ~4). Never quote a simulator
  projection as the estimate (ours overshot by 1.5-8 points).
- The test record lists every stage with its result and the open items. The team lead approves the submission.

## Things that went wrong before (check them)
- A notebook built from live code that imports something the Kaggle kit lacks (now caught by the builder).
- An admission gate that reopened under preemption (only a long replica showed it).
- A host-memory pool size with a capacity cliff (only a heavy late-phase test showed it).
- Kaggle gives 2×T4 instead of 4×L4 unless the competition is attached to the notebook.
- `NCCL_P2P_LEVEL=SYS` is ~12× slower on Kaggle; GPU util 0.97 runs out of memory within an hour.
- Stopped Vast instances lose their GPUs; keep a box running if more tests may follow.
- Never kill processes by name on shared hosts.
