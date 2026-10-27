# E055 Kaggle replica toolkit

Purpose: run the dsv41 solver on a rented 4xL4 pod so that the run measures the real bottlenecks (deadline coverage,
throughput / KV, crashes, accuracy) and scores like the hidden Kaggle rerun (240 tasks, ~335 test pairs, 12 h). The
approach is measurement, not a fitted simulator: a hidden-sized workload, Kaggle's deadlines, Kaggle's KV pool, and a
report that reads every log the pipeline writes. The one remaining hardware gap (pods are faster than Kaggle's 50 W
L4s) is measured on a Kaggle / pod twin pair and removed after the run.

| file | what it does |
|---|---|
| `make_workload.py` | builds a hidden-sized workload: the 120 eval tasks as they are, plus one transformed copy each (D8 transform + permutation of colours 1-9), seeded, validated |
| `workloads/evalx2t_s2026_*` | the default workload: 240 tasks / 344 pairs, seed 2026 (`challenges`, `solutions`, `idmap`, `manifest` with sha256) |
| `replica_run.sh` | launches a replica through the pod's `/root/podrun.sh` with the fidelity corrections, and writes `fidelity.md` + `replica.json` |
| `replica_report.py` | one markdown report per run dir: score, coverage, throughput, time decomposition, memory, bottleneck verdict |
| `calibrate_twin.py` | derives the Kaggle-vs-pod constants (k, d) from a twin pair and checks the stretch against Kaggle's real token count |
| `test_toolkit.py` | 13 tests (CPU, ~2 s): `.venv/bin/python -m unittest experiments/E055_kaggle_replica/toolkit/test_toolkit.py` |
| `reports/` | reports of the archived runs (see the findings below); `snapshot_*` are 2 h snapshots of the two live replicas |

## 1. Workload (`make_workload.py`)

```bash
.venv/bin/python experiments/E055_kaggle_replica/toolkit/make_workload.py              # seed 2026 -> workloads/evalx2t_s2026_*
.venv/bin/python experiments/E055_kaggle_replica/toolkit/make_workload.py --seed 7      # another draw of transforms
.venv/bin/python experiments/E055_kaggle_replica/toolkit/make_workload.py --check evalx2t_s2026   # re-validate
```

- **Why transformed copies.** The private set has ~335 pairs over 240 tasks; the eval set has 120 tasks. The first
  replicas used `../eval_x2_*`: the same 120 tasks twice under new ids. The solver orders tasks by size, so a copy runs
  next to its original. Its prompt is then identical, so vLLM serves the copy's first prompt from the cache: ~7.1k
  tokens, 15% of an attempt's final context (v5_w36 traces). The two streams also share those KV blocks. That makes
  prefill and KV optimistic: about 0.85M prefill tokens saved, and up to ~10-15% less KV while the pair is live.
- **Transform.** Each copy gets one of the 7 non-identity D8 elements and a permutation of colours 1-9 (0 stays the
  background). The permutation moves every colour the task uses. The same transform is applied to every grid of the
  task and to its solutions.
- **Seeding.** `random.Random(f"{seed}:{task}:{k}")` is independent of task order. Building twice gives byte-identical
  files (sha256 in the manifest).
- **Prompt divergence.** Each copy is drawn up to 64 times until its prompt (the harness's own `build_prompt`)
  diverges from the original's inside the first row of the first grid.
  - 115/120 copies diverge there; the median shared length is 2 characters (`[[`).
  - The 5 others are grids whose four borders are background in every D8 image; the worst shares 144 characters, about
    one row of zeros, less than one 128-token KV block.
- **Validation.** The build fails on any error. It checks that:
  - every copy grid inverts to the original (2,120 grids, solutions included);
  - shapes come out right: (h, w) or (w, h) depending on the transform; 238 grids change shape;
  - one solution exists per test input;
  - ids are unique and unused by any ARC file in `data/`;
  - no copy equals another task.
- **What the copies are not.** A copy is a new prompt for the same puzzle, so outcomes of a task and its copy are
  correlated. The report's bootstrap resamples original tasks (a task and its copy together).
- **Untested:** whether E128 finds rotated or recoloured puzzles harder. The report splits accuracy and score by copy
  vs original. (The live `eval_x2` copies are untransformed, so they cannot answer this.)

## 2. Launch (`replica_run.sh`)

```bash
T=experiments/E055_kaggle_replica/toolkit
# next candidate (multi-test + budget 0.5 + context limit, x44 + gate 0.7, retention 512, new code + auto-repair)
$T/replica_run.sh <port> <ip> cand_t1 evalx2t_s2026 ARC_WORKERS=44 ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 ARC_KV_ADMIT_MAX=0.7 ARC_KV_ADMIT_MIN_WORKERS=16 ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512"]'
# calibration: V5 = V5's Kaggle kit, per pair, x36, no gate (podrun's default server args are V5's: Humming, graph cap 64)
$T/replica_run.sh <port> <ip> v5cal_t1 evalx2t_s2026 ARC_KIT=/root/kit_v5 ARC_WORKERS=36
# calibration: v3 = the same kit at x48 with the solver's own server defaults (max-num-seqs 128, no Humming / graph
# cap) and Kaggle's v3 KV pool of 1.44M: choose <u> so that the report's "KV pool" line reads ~1.44M (with Humming,
# pods give 1.403M at 0.928 and 1.557M at 0.938; the default dense backend differs, so check the first minutes)
$T/replica_run.sh <port> <ip> v3cal_t1 evalx2t_s2026 ARC_KIT=/root/kit_v5 ARC_WORKERS=48 ARC_MAX_NUM_SEQS=128 \
  'ARC_SERVER_EXTRA_ARGS=[]' ARC_GPU_UTIL=<u>
REPLICA_DRY_RUN=1 $T/replica_run.sh ...    # print every remote command, touch nothing
```

`/root/kit_v5` (V5's Kaggle kit, hashes checked) exists on pod wd44jicex44qyc. On other pods, unpack
`kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz` there first.

Corrections it applies:
- **Deadlines 600 s earlier.** On the pod: stop-new 10.67 h, force 11.00 h, hard stop 11.33 h, kill switch 11.58 h.
  - Why 600 s: the solver's clock T0 is the notebook's first cell (`ARC_T0` in `local_run.py` cell 0). Deadlines come
    from `ARC_STOP_NEW_S` / `ARC_FORCE_AT_S` / `ARC_HARD_STOP_S` (solver) and `ARC_KILL_AT_S` (notebook kill switch).
    On Kaggle three steps follow T0: the offline install (256 s), the toolchain check (17 s) and the FUSE model load
    (590 s vs ~285 s on a pod). So Kaggle's server comes up at +0.24 h; pods reach that point at +0.06-0.10 h.
  - A deadline you pass is read as Kaggle-clock and shifted too.
- **KV pool.** `ARC_GPU_UTIL=0.938` gives 1,556,850 tokens on pods, against Kaggle's 1,554,183 at 0.945.
- **Logging and records.**
  - The workload is copied with a sha256 check.
  - A memlog records cgroup `memory.current` (what Kaggle's 173 GiB limit applies to; the pod's `free -g` shows the
    whole host), RSS, and per-GPU power and SM clock.
  - The script writes `runs/<name>/fidelity.md` and `replica.json` locally and copies both into the pod's run dir.
- **Safety.**
  - The script never stops or kills anything. It refuses to start when a solver is running, a GPU holds more than
    1 GiB, or the run dir exists.
  - `REPLICA_POWER_W=50` tries `nvidia-smi -pl 50`. It is opt-in, records the result, and Runpod containers usually
    lack the permission.

## 3. Report (`replica_report.py`)

```bash
rsync -a --exclude scratch -e "ssh -p <port>" root@<ip>:/root/runs/<name>/ experiments/E055_kaggle_replica/toolkit/runs/<name>/
.venv/bin/python experiments/E055_kaggle_replica/toolkit/replica_report.py experiments/E055_kaggle_replica/toolkit/runs/<name> --out .../report.md
# options: --solutions / --challenges / --idmap (else from replica.json or env.txt), --baseline <run dir> (paired
# difference by task), --no-traces (skip dsv41_traces.tar.gz)
```

Layouts it reads: pod runs, E050 runs (no `out/`), Kaggle bench output (`kernel_log.json` in place of `run.log`), and
live mirrors (flat `attempts.jsonl` / `metrics.jsonl` / `solver.log` / `status.json`; the submission is then rebuilt
from the records the way the solver ranks answers). Reading the sections:

- **(a) Score**
  - **Kaggle's leaderboard metric is task-weighted.** Each task scores the fraction of its test pairs answered, and the
    tasks are averaged. V5's 58.19 = 139.67/240 and v3's 50.97 = 122.33/240 are exact thirds, and no per-pair
    denominator between 100 and 500 reproduces both. Per-pair scores (`local score: x/172`, `multi_sim.py`) run ~1-3
    points below task-weighted, because the weaker multi-input tasks weigh more per pair.
  - **Cluster-bootstrap sd** over original tasks.
  - **Per-attempt accuracy** by attempt, train_pass, deadline cut, copy vs original.
  - **Answer-path losses:** a correct finished grid missing from the submission. This should be 0, and is 0 in every
    archived run.
  - **Kaggle-clock correction (the score to compare with the leaderboard).**
    - The pod timeline is stretched by (1 + (k - 1) s) / d per 10-s server interval, where s is that interval's prefill
      share of server time. The ~600 s staging gap is added before server-up.
    - Attempts that would start after stop-new or end after the force deadline on that clock are dropped, and the
      submission is rebuilt from what remains.
    - Three clocks are shown: central (k 1.45, d 0.94), optimistic (1.3, 1.0), pessimistic (1.6, 0.94).
    - It applies only to L4 runs that reached Kaggle's force deadline on the stretched clock.
- **(b) Coverage**
  - Jobs (attempt units: one per pair, or one per multi-test task) without a finished attempt 0; attempts per job and
    per task; second attempts.
  - Force-deadline and loop cuts; errored records (re-queued vs final); admission-closed time; last finished attempt.
- **(c) Throughput**, from `metrics.jsonl` (5-min windows) and `server.log` (10-s windows):
  - per-hour gen tok/s, prefix hits, uncached tok/s, KV mean / max, running (solver vs server), preemptions,
    attempts done, thrash-signature windows (hits < 0.85 and uncached > 500 tok/s);
  - the **prefill share of server time**: decode-stall vs clean-decode windows at the same batch size. vLLM counts a
    prompt's tokens only when its prefill completes, so "prompt 0" windows inside long prefills are excluded by taking
    the 90th percentile.
- **(d) Time decomposition**
  - Worker-slot time: in the server / outside it (tools, train check, harness) / idle (gate, refill).
  - Per attempt: decode-equivalent time (completion tokens / the per-stream rate over its lifetime) vs wall time, plus
    the tool-call statuses in the traces.
  - KV gate hold episodes, server deaths and downtime, startup allocator warnings vs OOM errors during the run, solver
    re-queue / retry / loop errors.
  - **Full re-prefills** (turn > 1, prompt >= 50k, cached <= 640) from the traces.
- **(e) Memory**
  - Host RSS and cgroup per hour, peak vs 173 GiB, a slope extrapolated to 11.75 h, and GPU power / clock.
- **(f) Verdict**, in task-weighted points:
  - **coverage (deadline):** jobs without attempt 0 x the accuracy of late attempt 0s, plus jobs without attempt 1 x
    the second-attempt gain;
  - **throughput/KV:** tokens below the run's healthy rate before stop-new, by cause (thrash, gate hold, prefill
    burst, low concurrency), converted to attempts and then to points;
  - **crashes:** downtime, tokens of errored attempts, and jobs lost to a final error;
  - **accuracy:** attempted pairs without a correct answer. This is the ceiling for model and harness work.
  - Coverage is what the deadline left undone; throughput and crashes are causes, not additive with it. A run stopped
    before its force deadline gets no coverage verdict.

## 4. What "trustworthy" means here

A replica is trusted when the configurations with leaderboard scores reproduce those scores within the replica's
draw spread, using the same workload, the same launcher corrections and the Kaggle-clock central score:

| config | leaderboard | replica (Kaggle clock, central) | status |
|---|---|---|---|
| V5 (per pair, x36, no gate, Humming) | **58.19** | to run: `v5cal_*` | the live `v5rep_x2` uses untransformed copies and unshifted deadlines: its raw score is optimistic; read its Kaggle-clock score, not the raw one |
| v3 (per pair, x48, KV 1.44M) | **50.97** | to run: `v3cal_*` | |
| V6 (multi-test, x36, gate 0.6) | pending | to run after its score lands | |

Acceptance rule:
1. Each calibration replica is within 2 sd of its leaderboard score. The sd of the difference is about
   sqrt(sd_replica^2 + sd_LB^2) ≈ sqrt(4^2 + 3^2) ≈ 5 points: the cluster-bootstrap sd is ~4 on 120 clusters, and the
   leaderboard's own attempt noise over 240 tasks is ~3.
2. The paired difference V5 - v3 on the same workload (`--baseline`) is within 2 paired sd of the leaderboard's +7.2.
   This is the sharper test, because it checks the config-dependent mechanisms (KV, thrash, coverage) that a replica
   exists to measure.
3. When both pass, but the replicas sit on the same side of the leaderboard by more than 1 sd, record that offset as
   the workload's bias (eval vs private set) and subtract it from new candidates.

Even then, one replica can only detect gross biases (more than ~10 points). Rank candidates by paired comparisons on the
same workload and seed (`--baseline`), and run 2-3 draws (`make_workload.py --seed`) when two candidates are within
noise.

## 5. Findings on the archived runs (`reports/`)

None of the archived runs is a full 12 h hidden-sized replica. Most were stopped by an operator signal after 1-5 h. So
the coverage verdict is only measured on the deadline tests; for the rest the reports give throughput, KV, crash and
accuracy facts.

| run | end | task-weighted (sd) | per pair | prefill share | max KV | verdict (largest infrastructure loss) |
|---|---|---|---|---|---|---|
| mt44_g70_ri512_new (candidate config, eval) | signal 5.14 h | 67.78 (4.1) | 65.1 | 20% | 0.89 | throughput/KV 2.1 pts (gate hold 0.45M + thrash 0.27M tokens; 4 thrash windows at 3.7-4.9 h); accuracy 23.9 |
| mt44_b05_ri512 (x44 ungated) | signal 2.93 h | 32.08 (4.3) | 30.8 | 37% (76% in hour 2) | 0.97 | throughput/KV 3.7 pts in 1 h of thrash (9 windows, 48-95 tok/s, 21 full re-prefills = 37% of uncached tokens) |
| v5_w36 (V5 config, eval) | signal 4.79 h | 48.75 (4.4) | 46.5 | 14% | 0.71 | throughput/KV 0.2: clean (KV <= 0.71, 0 thrash, 0 re-prefills) |
| c36_deadline (deadline test, force 1.17 h) | finished 1.17 h | 20.97 (3.7); **Kaggle clock 14.72** (17.22 / 14.72) | 19.2 | 13% | 0.40 | coverage 52.7 (73 jobs without attempt 0); the Kaggle clock keeps 21 of 47 attempts |
| eff50 (E050, RTX 4090, 1 attempt) | all jobs done 2.75 h | 71.39 (3.9) | 69.8 | 19% | 0.66 | throughput/KV 0.6; Kaggle clock skipped (not L4) |
| c36_restart (server killed) | signal 1.75 h | 27.50 (4.1) | 26.7 | 17% | 0.50 | crashes 3.1 pts: 1 death, 3 min down, 36 re-queued attempts (0.57M tokens lost, downtime included) |
| mt44_g70_ri512_u970 (util 0.97) | signal 1.27 h | 11.67 (3.0) | 11.0 | 27% | 0.39 | **crashes 35.5 pts**: 2 Triton OOM deaths. `EngineDeadError` is not in the solver's `CONN_ERRORS`, so each death spends one of `ARC_JOB_TRIES` (2) for every in-flight job; after the second, 43 jobs are lost for good |
| mt36_b05_gate60 (V6 settings, pod twin) | signal 4.08 h | 56.81 (4.3) | 55.2 | 16% | 0.71 | throughput/KV 0.9 |
| kaggle_bench_v1 (V6 settings, Kaggle, force 3 h) | finished 3.00 h | 39.03 (4.4) | 37.8 | 15% | 0.76 | coverage 32.4 (bench deadline); KV gate held at 1.96-2.40 h |

Cross-checks:
- The scorer matches the solver's own `local score` line on every run that printed one (33/172, 120/172, 65/172).
- Tools take only ~0.3-1% of slot time: decode-equivalent time is 93-99% of attempt wall time.
- Host RSS stays at 105-115 GiB, against 173 GiB.

On the stretched clock (central; extrapolated at each run's mean stretch past its end), Kaggle's force deadline
falls at pod time 8.9-9.8 h for these L4 configs (up to 10.4 h on the optimistic clock). So a pod run with Kaggle's default deadlines gets **~1.4-2.3 h more attempt time**
than Kaggle; the 600 s shift alone covers only 0.17 h of it. The live `v5rep_x2` and `cand_x2` replicas gain
~1.2-1.4 h this way (approximate: their mirrors lack `server.log`), besides the shared-prefix optimism of their
workload.

## 6. Known gaps (what still limits fidelity)

1. **The throughput correction rests on one twin pair, in-sample.**
   - Kaggle bench V1 vs pod `mt36_b05_gate60`: 3 h of Kaggle data at V6 settings.
   - Effective prefill 821 vs 1,187 tok/s (k = 1.45); pure decode 9.60 vs 10.20 tok/s per stream (d = 0.94).
   - Kaggle generated 0.89 / 0.89 / 0.91 x the pod's tokens 1 / 1.5 / 2 h after server-up; the stretch predicts
     0.88 / 0.88 / 0.89 (`calibrate_twin.py`).
   - Untested: late hours, thrash (s up to 0.75), and per-pair configs. A second Kaggle bench at another config would
     validate the correction out of sample.
2. **Eval vs private set.** A heavier or harder private set moves KV pressure and accuracy together; the calibration
   offsets of section 4 are the only handle on it.
3. **Restarts on Kaggle** reload from the FUSE page cache, which is untested. The stretch does not lengthen outages.
4. **Determinism.** The stack is not run-to-run deterministic, so every replica is one draw.
5. **Power cap.** Untested whether Runpod lets `nvidia-smi -pl 50` through; if it does, the hardware gap closes at the
   source.

## Note on toolkit/runs/
Live mirrors used for reports during runs; finished runs are archived in `../runs/<name>/` (with traces).
