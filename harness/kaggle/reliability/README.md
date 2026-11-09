# Reliability kit for Kaggle submission candidates

Our own kit to qualify a candidate notebook (a vLLM-served LLM on Kaggle's 4xL4, ~12 h competition rerun) on a
rented 4xL4 pod, before it spends a daily slot. It turns tonight's ad-hoc pod scripts (E062 `pod3_scripts/`,
`phase_h/scripts/`) into tested tools and encodes the measurement pitfalls they found (see [Pitfalls](#pitfalls)).

Never run any of it on Kaggle (the tools refuse inside a Kaggle kernel). Every Kaggle run still needs the team
lead's OK ([../SUBMISSION_TESTING_PROTOCOL.md](../SUBMISSION_TESTING_PROTOCOL.md) stage 6).

| file | what it does |
|---|---|
| `run_notebook.py` | runs the exact notebook end to end with nbclient (Kaggle's "Run All"): `/kaggle/input` mapping by symlinks, a fresh working dir, `--mode commit` / `--mode rerun` (`KAGGLE_IS_COMPETITION_RERUN` absent / `=1`), `env -i`-like clean environment, live cell-output log, hooks at cell starts |
| `sampler.py` | read-only monitor, one JSON line per minute (every 5 s while the server is down or after a fault): container RAM (cgroup v1/v2) and Kaggle-eq, processes by group, GPUs, vLLM per-rank allocator lines, solver / supervisor / attempts counters |
| `watcher.py` | threshold alerts over the sampler's JSONL, to a file and stdout; `--follow` loop or `--once` polls (exit code = level) |
| `report.py` | snapshots, per-hour dynamics, slopes after warm-up, time to limit, events, fault recovery, and a one-line verdict |
| `gpu_hold.py` | holds a calibrated amount of memory on every GPU so the pod's vLLM headroom equals Kaggle's; `calibrate` computes the size |
| `fault_inject.py` | `kill-tp`, `sigstop-engine` / `sigcont-engine`, `gpu-oom-burst`, `cpu-stress`, `find`, `verify`; exact targets only, `--dry-run` |
| `recovery.py`, `reli_common.py` | recovery checks; shared parsers and trend maths (stdlib only, Python >= 3.8) |
| `tests/` | `.venv/bin/python -m unittest discover -s kaggle/reliability/tests` (no GPU; synthetic /proc, cgroup and log fixtures; decoy processes). `RELI_NB_PYTHON=<python with nbclient + ipykernel>` adds two live notebook runs |

Everything except `run_notebook.py` is stdlib-only and runs under the pods' `/usr/bin/python3`.
`run_notebook.py` runs under the kernel venv's Python, which needs `nbclient` and `ipykernel`.
Copy the folder to the pod, e.g. `scp -r kaggle/reliability root@pod:/root/kit`. All clocks are UTC.

## Qualification protocol

A candidate qualifies when all four steps pass. Write the results into the candidate's test record
(`../SUBMISSION_TESTING_PROTOCOL.md` stage 3 and 5).

### Step 0. Pod and inputs
- A 4xL4 pod with >= 256 GB host RAM and >= 200 GB disk.
  The install cells write a ~20 GB uv cache plus a ~20 GB venv; a full disk fails cell 1's `dpkg` (phase H).
- The exact frozen notebook (sha256 recorded) and its exact input versions: the datasets as they were at the build,
  and the model verified per file against the notebook's own manifest.
- System packages that Kaggle's image has and the pod lacks (the cu13 debs' dependencies): see
  `experiments/E062_manh_vllm_fork/per_layer/phase_h/README.md`, "Second attempt".

### Step 1. Commit-mode smoke (~10 min)
Start a sampler for the run, then run the notebook in commit mode, exactly as Kaggle's commit run does:
```bash
K=/root/kit; R=/root/runs/<cand>_commit; W=/root/kwork_<cand>_commit; mkdir -p $R
setsid nohup /usr/bin/python3 $K/sampler.py --out $R/mon.jsonl --working $W --timeline $R/nb.timeline \
  --stop-on-notebook-end > $R/sampler.out 2>&1 < /dev/null & echo $! > $R/sampler.pid
/root/kenv/bin/python -I $K/run_notebook.py NB.ipynb --out $R/nb --mode commit --clean-env --venv /root/kenv \
  --map /root/models/<m>=models/<owner>/<slug>/pytorch/default/1 --map /root/comp=competitions/arc-prize-2026-arc-agi-2 \
  --map /root/ds/<name>/<ver>=datasets/<owner>/<name> --prune-input --working $W --dry-run   # check, then drop --dry-run
```
- Use one `--map` per Kaggle input. `--prune-input` removes stale symlinks, so the notebook's glob finds one model
  dir, as on Kaggle.
- Reset the kernel venv to a pristine copy before each run (`rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv`):
  the notebook's install cells install into it.
- **Pass:**
  - `nb.timeline` ends with `NOTEBOOK_DONE ok`;
  - the notebook's own checks print OK (e.g. `CHECK OK`, `SANDBOX CHECK OK`, contract `fail: []`, `INPUT INTEGRITY OK`);
  - the KV pool line equals the expected one;
  - `report.py` shows no OOM, allocator retry or server death.
- Keep the server log: its `Capped the PyTorch allocator` lines are the pod's baseline for step 2.

### Step 2. Kaggle-like GPU headroom (calibrate once per stack)
On a pod, vLLM finds more free VRAM than on Kaggle, so it gets a larger allocator cap. Without a hold, the pod
replica has ~0.5 GiB/GPU more headroom and can pass where Kaggle OOMs. Calibrate the hold from Kaggle's own numbers:
the `Capped the PyTorch allocator at X GiB (Y GiB reserved)` lines of the same stack's Kaggle smoke log (cap - reserved
per rank).
```bash
python3 $K/gpu_hold.py calibrate --hold-mib 0 --pod-log $W/dsv41_logs/server/server.log --kaggle-log kaggle_smoke_server.log
# or --kaggle 0.67-0.73 / --kaggle TP0=0.73,TP1=0.71,TP2=0.67,TP3=0.67
```
- The hold moves the cap one-for-one: B32-M went 410 -> 350 MiB and cap - reserved 0.61-0.66 -> 0.66-0.72 GiB,
  against Kaggle's 0.67-0.73.
- Accept when every rank is within ~0.02 GiB of Kaggle.
- Check that util x total still fits: vLLM refuses to start when free < util x total.

### Step 3. 12 h rerun-mode replica with monitoring (the stability and score record)
```bash
R=/root/runs/<cand>_rerun; W=/root/kwork_<cand>_rerun; mkdir -p $R      # test file mapped to the 240-task workload
setsid nohup /usr/bin/python3 $K/sampler.py --out $R/mon.jsonl --working $W --timeline $R/nb.timeline \
  --fault-log $R/fault.jsonl --stop-on-notebook-end > $R/sampler.out 2>&1 < /dev/null & echo $! > $R/sampler.pid
setsid nohup /usr/bin/python3 $K/watcher.py --samples $R/mon.jsonl --follow > $R/watcher.out 2>&1 < /dev/null &
setsid nohup /root/kenv/bin/python -I $K/run_notebook.py NB.ipynb --out $R/nb --mode rerun --clean-env \
  --venv /root/kenv --map ... --prune-input --working $W \
  --env 'ARC_SERVER_ENV_JSON={<the notebook value>, "VLLM_GPU_MEMORY_LOG_INTERVAL": "60"}' \
  --on-cell-start "<cell before the server cell>:/usr/bin/python3 $K/gpu_hold.py hold 350 --log $R/gpu_hold.log --pid-file $R/gpu_hold.pid" \
  > $R/runner.out 2>&1 < /dev/null &
```
- **Deviations from the exact notebook** (record them):
  - the GPU hold;
  - `VLLM_GPU_MEMORY_LOG_INTERVAL=60`. Logging only: the server then prints the per-rank `GPU memory:` line every
    60 s. Our dsv41 notebooks read the server env from `ARC_SERVER_ENV_JSON` through `setdefault`, so the env value
    wins; other notebooks need their own hook.
- Start the hold before the cell that starts the server (`--dry-run` lists cell indices). Check `gpu_hold.log` and
  the server's cap lines against step 2.
- Monitor from your workstation without touching the pod's processes:
  - poll with `ssh pod python3 $K/watcher.py --samples $R/mon.jsonl --once --summary --state $R/poll.json --alerts $R/poll.alerts`
    (exit code 0 = nothing new, 2 WARN, 3 ALERT, 4 CRIT);
  - or wait for a wake-up with `watcher.py --follow --exit-on ALERT --until 04:00`, which prints `FLAG ...` or
    `HOURLY_DUE` plus a status summary.
- Hourly, copy `mon.jsonl` and `dsv41_logs` (without traces) and run `report.py` locally.
- **Pass:**
  - verdict `OK` (or `WATCH` with a stated reason);
  - Kaggle-eq peak + ~20 GiB margin under 173 GiB, with no growth toward it;
  - worst cap - peak never under 0.26 GiB and not shrinking toward it before the run's end;
  - 0 allocator retries and 0 OOMs; no unplanned server death; 0 attempts finalized as error;
  - never two engines; the notebook ends `ok` with a valid submission.
- Plus the throughput, coverage and accuracy criteria of the protocol's stage 3.
- Stop a run only by pid / process group (`kill -- -$(ps -o pgid= -p <pid>)` after checking the cmdline), never by name.

### Step 4. Fault injection (a separate ~3 h rerun-mode run, same setup as step 3)
Inject under real load: >= 10 min after the server is up, with all workers running. Give the sampler the same
`--fault-log`, so it samples every 5 s for 15 min after each action. Check the candidate's restart budget first,
e.g. our cell 2s supervisor allows 2 restarts and only with >= 60 min left. Space the actions so that each recovery
finishes before the next one. Do not spend the budget in the 12 h record run.
```bash
python3 $K/fault_inject.py find                                   # servers, worker / engine titles, GPU apps
python3 $K/fault_inject.py kill-tp 3 --log $R/fault.jsonl --dry-run    # check the target, then without --dry-run
```
| test | command | what must happen |
|---|---|---|
| worker crash | `kill-tp 3` (~+20 min) | the API server exits; the supervisor kills the group, waits for the GPUs to be empty, relaunches; the solver re-queues in-flight jobs |
| GPU OOM | `gpu-oom-burst 1 --hold-s 60` (~+60 min) | either absorbed (retries rise, no death) or an engine death that recovers like the crash |
| hang | `sigstop-engine` (~+90 min; no `--resume-after`) | the supervisor's stall detection (25 min without tokens while requests run) kills and relaunches |
| CPU contention | `cpu-stress --procs $(( $(nproc) / 2 )) --seconds 1800` | no death, no finalized error; the throughput dip is reported |

- **Pass criteria**, from `fault_inject.py verify --log $R/fault.jsonl --samples $R/mon.jsonl --logs-dir $W/dsv41_logs`
  (also in `report.py --fault-log`):

  | check | pass | fail |
  |---|---|---|
  | finalized_errors | no attempt finalized as error after the fault (`stop=error`) | any; one was lost in run 3 |
  | server_died | the solver logged the death (kill, hang) | the fault did not register / the hang was never detected |
  | requeued | `re-queue` lines after the death | in-flight jobs dropped |
  | tries_spent | no `retry` lines | WARN: a retry spends one of the job's tries, so a second death finalizes those jobs as errors (run 3: 64) |
  | one_engine | never two EngineCores, no old/new vLLM overlap, Kaggle-eq peak <= pre-fault + 10 GiB | two engines' host RAM: an OOM-kill on Kaggle |
  | ram_limit | Kaggle-eq < 173 GiB throughout | over |
  | restart_time | `server up` <= 900 s after a kill (hang: stall limit 2100 s + 900 s) | slower, or never |
  | jobs | errored jobs finished normally later | WARN while pending, lost if the run ended |
- Exit code 0 = no FAIL. Record the timeline lines: old processes gone, GPUs free, first new process, supervisor
  states, solver death and up lines.
- Resume a stopped engine by hand only if the supervisor failed: `fault_inject.py sigcont-engine`.
- **Targeting safety:**
  - exactly one process with the exact title (`VLLM::Worker_TP<r>`, with `_DCP<n>` style suffixes) inside the one
    vLLM API server's tree, or `--server-pid`;
  - the cmdline is re-read through a pidfd right before the signal;
  - `--proc-root` other than `/proc` is dry-run only.

### Step 5. Report and record
```bash
python3 kaggle/reliability/report.py --samples copy/mon.jsonl --logs-dir copy/dsv41_logs --fault-log copy/fault.jsonl --json copy/report.json
```
Put the `VERDICT:` line, the trend table and the fault checks in the test record. Archive the run copy (mon.jsonl,
alerts, fault.jsonl, logs without traces) in the experiment folder.

## Alerts (watcher.py)

| code | level | meaning and action |
|---|---|---|
| `ram` | WARN 155 / ALERT 165 / CRIT 173 GiB | Kaggle-eq host RAM. CRIT = Kaggle would OOM-kill the notebook. Look at the groups (TP workers, sandboxes, EngineCore) for the grower |
| `ram_source` | WARN | no cgroup accounting readable: the figure is host-wide `/proc/meminfo` and overstates |
| `cap_peak` | WARN < 0.5 / ALERT < 0.26 GiB | worst rank's allocator cap - peak allocated. 0.26 is the OOM band of the B32-M audit. Read the trend: shrinking toward 0.26 before the end = the candidate OOMs on Kaggle |
| `retries` | ALERT | a rank's `allocator retries` grew: the allocator reached its cap and flushed its cache. The first real pressure signal |
| `vllm_oom` | CRIT | a rank's OOM count grew or a `CUDA out of memory` line appeared |
| `engine_dead` / `traceback` | ALERT / WARN | new EngineDeadError / worker-died / execute_model-timeout lines / Tracebacks in the server logs |
| `two_engines` | CRIT | > 1 EngineCore: a restart holding two engines' RAM at once |
| `server_died`, `supervisor_restart`, `not_up` | ALERT | the solver saw a death; the supervisor's start count changed; its state is not `up` for > 5 min |
| `stop_error` | ALERT | an attempt was finalized as error (lost). Per-try error records of re-queued jobs are normal |
| `error_burst` | WARN | >= 5 new error records in 10 min |
| `timeout` | WARN | APITimeoutError / `Request timed out`: a try spent without a server death |
| `trend` | WARN | a slope after warm-up reaches its limit before `--end-h` (re-sent at most hourly) |
| `stale` | ALERT | no new sample for > 300 s: the sampler died or the host is stuck |
| `notebook_end` | INFO | the runner logged the end; `--follow` exits |
| `smi_free` | INFO, off by default | nvidia-smi free under `--smi-free-warn-mib`; board level, not a pressure signal |

Thresholds are CLI flags with env defaults: `RELI_RAM_WARN/ALERT/CRIT`, `RELI_CAP_PEAK_WARN/ALERT`,
`RELI_ERROR_BURST`, `RELI_ERROR_BURST_MIN`, `RELI_NOT_UP_MIN`, `RELI_STALE_S`, `RELI_SMI_FREE_WARN_MIB`,
`RELI_END_H` (11.5, the solver's hard stop on the Kaggle clock), `RELI_WARMUP_MIN` (20), `RELI_WINDOW_H` (2),
`RELI_TREND_EVERY_MIN` (30). Level crossings fire once per level (per server instance for VRAM).

## Reading the dynamics

One snapshot says little: in the first hour every number looks fine. The report shows how the numbers change.
- **Clock.** Hours on the Kaggle clock (the solver's elapsed time from its log, t0 = notebook start), so ETAs
  compare directly with the solver's deadlines (stop-new 10.83 h, force 11.17 h, hard stop 11.5 h).
- **Warm-up and segments.**
  - Slopes skip the first 20 min after each server start: the model load and the KV and CUDA-graph allocation.
  - Each server instance is its own segment: a restart resets peaks and counters.
  - The trend uses the current server only, over the last 2 h by default.
- **Slope and time to limit.** Least-squares slope per hour; `ETA = (limit - now) / slope`.
  - `stable`: not moving toward the limit, or ETA > 48 h.
  - `growing toward X, ETA ~Y h (at Z h, before the end)`: RISK.
  - The same after the run's end: WATCH.
  - `blew up at Z`: FAIL; the first OOM, or Kaggle-eq >= 173.
- **Per-hour table.** The change within each hour, per hour. It shows deceleration that a single fit hides. Run 3's
  worker VRAM grew +200 MiB in hour 1 and +80 MiB in hour 3, so a linear ETA from early hours is conservative. Read
  both before calling a candidate.
- **Which VRAM number.**
  - Allocated flat, with peak and reserved creeping, is cache growth with the peak, not a leak.
  - What kills is cap - peak reaching the spike size (~0.26 GiB): see the v12 replica,
    `phase_h/manh_v12/README.md` report 2, where cap - peak fell 0.19 GiB/h for an ETA of ~7 h.
  - Cap - reserved reaching 0 only starts allocator retries, which alert at once.
- **Faults are not instability.** Deaths inside a fault window (fault time + 1 h) are marked `(fault injection)`
  and judged by the recovery checks. Any other death fails the verdict.

## Pitfalls

Found on 2026-10-29/30 (E062 phase H, pod 3 runs 2-3, B32-M v3 replica, Manh v12 replica). The kit handles each one.
1. **`/proc/meminfo` on pods is host-wide.**
   - At 03:56 it showed AnonPages 131.7 GiB against the container's cgroup total_rss of 120.4 GiB: 11-13 GiB
     belonged to other tenants, even at idle.
   - Kaggle-eq = cgroup anon + shmem + 7 GiB Kaggle overhead, against Kaggle's cgroup v2 `memory.max` of 173.0 GiB
     with no swap.
   - Pods run cgroup v1 (`memory/memory.stat` total_rss + total_shmem), Kaggle v2 (`memory.stat` anon + shmem).
     Detect which one, not assume it. A smoke check that assumed v2 failed on the pod with `max() iterable argument
     is empty`.
2. **nvidia-smi free is not a pressure signal.**
   - vLLM caps the PyTorch allocator just under the free memory it sees at startup ("keeping 0.25-0.5 GiB free for
     other GPU allocations"), so the board still shows free memory when the allocator hits its cap and an
     allocation fails.
   - Use the per-rank `GPU memory: ... allocator retries, ... OOMs` lines (enable with `VLLM_GPU_MEMORY_LOG_INTERVAL`)
     and the startup `Capped the PyTorch allocator at X (Y reserved)` line: track cap - peak and the retries.
3. **The pod has more GPU headroom than Kaggle.**
   - Kaggle had ~0.38 GiB more non-torch memory per GPU (CUDA graphs 0.65 vs 0.34 GiB).
   - B32-M passed on a pod and scored 46.5 on Kaggle against ~77.9 on the replica; the audit predicted a GPU OOM at
     ~3-4 h with Kaggle's VRAM.
   - Emulate the headroom with `gpu_hold.py`, calibrated per stack.
4. **Snapshots mislead.** Early windows overstate stability. Report slopes after warm-up and time to limit
   ([Reading the dynamics](#reading-the-dynamics)).
5. **Exact targets only.**
   - Worker titles are set with setproctitle: the cmdline is one token padded with NULs (`VLLM::Worker_TP3`, or
     `VLLM::Worker_TP0_DCP0` with DCP). A bare `VLLM::Worker` exists briefly at startup.
   - `pkill -f` patterns hit other agents' runs (E021 incident). Kill by pid after checking the cmdline.
   - The rank with the largest pinned Engram tables frees its GPU ~20-25 s after it exits. A restart must wait for
     the GPU process list to be empty, or two engines coexist.
6. **A dying server's HTTP 500 spends tries** (run 3, solver S26).
   - The 500's body does not say "EngineDeadError", so each in-flight job used a `retry` on the dying server.
     A second death would have finalized 64 jobs as errors.
   - `APITimeoutError` also cost a try. Two attempts were lost.
   - Hence the `tries_spent` and `timeout` checks.
7. **Per-try error records are not lost attempts.**
   - A re-queued job writes an error record per try. Lost means finalized: the solver's `stop=error` line, or a job
     whose error record has no later normal record at the end.
   - The first metrics window after a restart has negative rates (the counters reset). Ignore it.
8. **Runner fidelity.**
   - nbclient writes the executed notebook only at the end, so the 12 h solver cell's output (supervisor restarts)
     was not on disk mid-run. `run_notebook.py` streams outputs to `<out>.log` and rewrites `<out>.ipynb` after
     every cell.
   - Commit mode must not inherit `KAGGLE_IS_COMPETITION_RERUN`. The runner removes it.
   - Start the sampler on the run's own working dir: a probe started before the launcher re-pointed
     `/kaggle/working` read the previous run's server.log (phase H).
9. **Pod startup varies.**
   - Engram THP loading (MADV_COLLAPSE failures, memory compaction) took 495-1525 s to server up on the same pod,
     against 125 s on another. Judge startup against the notebook's own server timeout, not one pod run.
   - Pods run their L4s at 72 W, while Kaggle may run them at 50 W (unresolved), so pod throughput is optimistic.
