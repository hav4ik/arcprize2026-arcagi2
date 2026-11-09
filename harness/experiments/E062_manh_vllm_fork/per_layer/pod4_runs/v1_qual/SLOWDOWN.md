# Why V1 trailed V12 before the first fault (Kaggle 0-3 h)

2026-10-30. Read-only log analysis; no GPU or pod use. Question: why had V1 (b32m-fix V1, bec71fb1) answered 116 tasks at
3.0 h, against 126 for Manh's V12 on the same 240-task rerun workload?

## Answer
- **The cause is the pod, not the notebook.** V1 ran on pod 4. V12 and the control (SFT 025ep on V12) ran on pod 2.
  - The same V1 notebook (bec71fb1) also ran on pod 2 the night before (`../../phase_h/runs/r1_rerun_bec71fb1`, rerun
    mode, same workload, clean for 0.58 h until its kill test).
  - On pod 2 it started as fast as V12: cell 2m took 241 s, against 246 s.
  - On pod 2 it also decoded at V12's speed: 1.003x [0.996-1.010] at matched running streams and KV.
- **Pod 4 cost V1 in two ways:**
  1. **Startup:** the solver started 188 s later (+485 s against +297 s).
  2. **Speed:** at matched running streams and KV usage, decode was 6.3% slower, and the first-wave prefill was 16%
     slower.
- **Effect on the 10-task gap:** together these cost 0.36M generated tokens by 3 h (5.94M against 6.30M). When V12 had
  generated V1's 5.94M tokens, it had answered 120 tasks (the control: 122).
  - So throughput explains ~6 of the 10 tasks and 2.3 of the 4.7 score points.
  - The other ~4 tasks are completion-order noise. Per-task accuracy ties, and V12 and the control differ by up to 4
    tasks at equal tokens.
- **Our added layer has no measurable cost.** That layer is the cell 2s supervisor, solver 1ea2b25d, util 0.90,
  `--language-model-only` and the 900 s execute timeout.
- **The earlier "unmeasured layer" suspicion came from a cross-pod comparison.** The README's comparison put a pod-4 run
  against pod-2 runs.

## Runs
| | V1 (this run) | V12 | control | V1 on pod 2 |
|---|---|---|---|---|
| notebook | bec71fb1 | c47be5a8 | V12 + SFT 025ep model + g22df163d2d wheels | bec71fb1 |
| pod | **4**: 128 CPUs, 2 NUMA nodes, GPU3 SYS to the rest | **2**: EPYC 9254, 48 CPUs, PHB pairs, 1 NUMA node | 2 | 2 |
| window used | 0-3.0 h | 0-3.0 h | 0-3.0 h | 0-0.58 h clean; server data to 2.05 h (after a TP2 kill + restart) |

The server args are verified identical, except util 0.90 vs 0.91 and `--language-model-only` vs
`--limit-mm-per-prompt {"image": 16}`. The launch.json environment overrides are identical. V1's MANH_SERVER_ENV adds
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900. Both runners use the same `env -i` environment. V1 on pod 2 has exactly V1's
args.

## Evidence (0-3 h unless noted)
| metric | V1 pod 4 | V12 pod 2 | control pod 2 | V1 pod 2 |
|---|---|---|---|---|
| **Startup** | | | | |
| Manh install cell 4 | 59 s | 37 s | (install done +50 s) | 41 s |
| cell 2m (server) | **406 s** | 246 s | up at +280 s | **241 s** |
| - weights loaded, slowest rank | 105.6 s (TP3; TP0-2 50 s) | 46.5 s | - | 45.0 s |
| - init engine (compile, profile, KV, graphs) | 237 s (commit smoke on pod 4: 236 s) | 153 s | - | 150 s |
| solver start / first gen > 100 tok/s | +485 s / 0.159 h | +297 s / 0.099 h | +281 s / 0.094 h | +296 s / 0.098 h |
| first-wave prefill (64 prompts) | 3,450 tok/s | 4,130 | 4,050 | 4,080 |
| **Server (10 s engine lines; means while up)** | | | | |
| gen tok/s per half-hour | 669 703 577 489 524 528 | 735 729 554 563 540 531 | 743 723 549 609 520 557 | 743 (kill at 0.59 h) |
| running / server waiting | 43.4 / 2.1 | 42.3 / 4.4 | 43.5 / 2.4 | - |
| KV usage | 80.8% | 81.6% | 81.5% | - |
| GPU prefix hit / host (CPU-tier) hit | 84% / 50% | 67% / 65% | 85% / 51% | - |
| re-prefilled (uncached) / host-restored tok/s | 302 / 824 | 359 / 3,259 | 350 / 1,155 | - |
| preemptions | 0 | 0 | 0 | 0 |
| decode at 64 running, KV 30-80% | 740 tok/s = **86 ms/step** | 793 = 81 ms | 802 = 80 ms | 803 = **79 ms** |
| decode at 28-33 running, KV >= 86% | 514 = 60 ms | 533 = 56 ms | 538 = 57 ms | 544 = 55 ms |
| **decode vs V12 at matched running and KV** (lines) | **0.937** [0.933-0.943] (598) | 1 | 1.001 (658) | **1.003** [0.996-1.010] (410) |
| GPU SM clock / power / temp | 1474-1589 MHz / 72 W / 52-55 C | 1313-1367 / 72 / 67-70 | 1316-1369 / 72 / 67-70 | - / 72 / 67-70 |
| **Solver** | | | | |
| KV-gate holds (usage / server waiting) | 28 (7 / 21) | 29 (7 / 22) | 29 (9 / 20) | 1 by 1.08 h |
| first hold (h after first token) | 0.65 h (0.51) | 0.55 h (0.47) | 0.55 h (0.47) | - |
| solver in-flight minus server running+waiting | <= 1% | <= 1% | <= 1% | - |
| S25/S26 waits, re-queues, retries, errors before 3 h | 0 | 0 | 0 | - |
| finished by 3 h: wall p50, s per 1k out tokens, out tokens | 2,330 s, 81.6, 37.2k | 2,274 s, 79.1, 37.8k | 2,294 s, 75.4, 38.8k | - |
| **Outcome** | | | | |
| generated tokens by 0.5 / 3 h | 0.88M / **5.94M** | 1.10M / 6.30M | 1.13M / 6.43M | 1.11M / - |
| tasks / score at 3 h | 116 / 35.7 | 126 / 40.4 | 128 / 40.3 | - |
| **tasks / score at 5.94M generated tokens** | 116 / 35.7 | **120 / 38.1** | **122 / 38.2** | - |
| tokens in unfinished attempts at 3 h | 1.48M | 1.23M | - | - |

What the server table shows:
- **No extra prefill or eviction in V1.** V1 re-prefilled less than V12, restored less from the host tier and had 0
  preemptions.
- **No admission deficit.** V1 ran as many streams as V12 (more in 1-1.5 h, because its KV filled later and the gate
  bound later).
- **Tool time is not a factor.** Tool time is ~1% of attempt wall time in all three runs.

## Causes and their share of the 116-vs-126 gap
| cause | evidence | tokens by 3 h | tasks | score |
|---|---|---|---|---|
| pod 4 decodes ~6.5% slower per step | matched-load ratio 0.937 on pod 4; 1.003 for the same notebook on pod 2; also 0.931 / 0.951 / 0.928 against V12 + control in hours 1 / 2 / 3, so it is not a burst | -0.40M gross, +0.16M back because V1 ran slightly more streams: **-0.24M** | **~4** | **~1.5** |
| pod 4 starts slower | solver +188 s; first token +216 s; V12 at the same serving age had 6.18M | **-0.12M** | **~2** | **~0.8** |
| our layer (supervisor, solver diff, util, LM-only, timeout) | same notebook on pod 2: startup -5 s, decode 1.003x; solver logs show identical admission | ~0 | ~0 | ~0 |
| residual | at equal tokens V1 sits -5..+3 tasks from V12 / control. V12 vs control differ by up to 4 at equal tokens and by 7 at equal time (2 h: 87 vs 94). Per-task accuracy ties. V1 had 0.25M more tokens in unfinished attempts. | - | ~4 | ~2.4 |

The conversion uses V12's 0.36M tokens = 6 tasks, about 1 task per 60k tokens.

## What on pod 4 is slow (partly inferred)
- **Startup.**
  - **Compile:** init engine took 237 s on pod 4 against 150-153 s on pod 2. The pod-4 commit smoke gave the same
    (236 s), and a warm restart on pod 4 took 39 s. So the extra time is cold torch.compile/Triton work, which is
    CPU-bound. Torch reports 32 threads on pod 4 against 12 on pod 2.
  - **TP3 loading:** TP3 (GPU3, on the other NUMA node) finished loading 55 s after the other ranks. It is the only rank
    that logged "Engram MADV_COLLAPSE failed". It pins the largest Engram tables (30 GiB) on a host that had ~65 GiB free
    and ~530 GiB of page cache at the start.
- **Serving.**
  - **The bottleneck is host-side.** Pod 4's GPUs sit at the same 72 W cap but run at 150-250 MHz higher clocks. So per
    step they wait on the host or the interconnect, not on SMs.
  - **The same signal grows with the stall:** after the 3 h restart, pod 4 decoded 30% slower at matched load, and its
    clocks rose to 1,820-1,950 MHz.
  - **Candidates:**
    - Manh's host all-reduce (mode 556: CPU reducer threads summing bf16 in host memory), with GPU3 across the socket
      link. Large prefill messages were hit harder (-16%) than small decode messages (-6.5%).
    - Engram and embed_tokens UVA reads from pinned host memory.
    - A slower per-core CPU (pod 4's CPU model was not recorded; pod 3's 128-CPU host was an EPYC 7702).
    - Another tenant: ~86 GiB of host AnonPages were not in our cgroup.
  - The logs cannot separate these.

## Every V1-vs-V12 difference that could touch throughput or admission
| difference | effect before 3 h |
|---|---|
| Solver 1ea2b25d vs 11bfaaaa: the HTTP 500 string counts as a connection error. | none: there were 0 errors. |
| Solver: ARC_GRID_FORMAT (default json, unset in V1). | none |
| Solver: `start_external` waits while the status is not "up". | none: "server up after 0s". |
| Solver: the solver-owned `serve` command builder. | unused with an external server |
| Cell 1: ARC_CONN_TRIES 12 vs 4; MANH_VISION=0; the server cell's ARC_MAX_SERVER_RESTARTS 0 (the solver gets 3 in both). | none before a fault |
| Cell 0a GPU gate: commented out in V1, ~1 s in V12. | none |
| Cell 1b extra size check. | commit runs only |
| Cell 2m: util 0.90 vs 0.91 (KV 1.617M vs 1.609M); `--language-model-only` vs the vision tower; VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900. | none: the same args decode at 1.003x on pod 2 |
| Cell 2m: direct launch vs Manh's detached `--supervise`, which polls the process every 5 s and /v1/models only until the server is up. | none |
| Cell 2s supervisor (below). | none measurable |
| Cells 3s, 3w, 4, 4b. | after the solver |

**Cell 2s supervisor.** It runs as a daemon thread in the kernel process. Every 5 s it reads `/proc/<pid>/stat` and GETs
`/metrics` (timeout 10 s); the solver already reads `/metrics` every 10 s. It calls nvidia-smi only while stopping a
dead server.
- **Its GIL cannot stall the server or the solver.** Both run in their own sessions: the launcher's `Popen` with
  `start_new_session=True`, and cell 3's `Popen`. The kernel thread shares a process with neither.
- **Measured:** with this supervisor polling, the same notebook on pod 2 decoded at 1.003x V12.
- **nvidia-smi load is the same on both pods:** both pods' 60 s samplers call nvidia-smi.

## Still unexplained, and the cheapest experiments
- **Which pod-4 host property costs the 6.5%.** It matters only if pod-4-like hosts are used again, or as a hint for
  Kaggle. Kaggle's Xeon at 2.2 GHz is slower per core than pod 2's EPYC 9254, so pod-2 replicas may be optimistic if
  decode is host-sensitive.
  - **Cheapest experiment:** a 15 min fixed-load replay (`evals/serve_replay_bench.py`, 64 streams) on a pod-4-like
    host, with the server's processes default vs pinned to GPU-local NUMA nodes (`numactl`).
- **The ~4-task residual.** It needs no experiment to call it noise. If the team wants the 3 h task count of the
  notebook itself confirmed, the cheapest run is V1 fault-free for ~1.2 h on the pod that ran V12, compared at equal
  tokens.
- **The control does not control for the pod.** It reproduces V12 on the same pod 2.

## Side finding (after 3 h; changes the README's fault-cost reading)
- **Server 1 on pod 4 (after the TP3 kill) decoded at 0.70x V12 at matched running and KV** (876 lines over 3.05-6.0 h).
  - It also ran 25-30% slower than pod 4's own server 0 at the same load (e.g. 404 vs 514 tok/s at 31 running, KV 88%).
  - Its GPU clocks rose to 1,820-1,950 MHz.
  - The fast sampler was on only at 2.93-3.26 h, so it is not the cause.
- **On pod 2, V1's server after its TP2 kill decoded at 1.008x V12.**
- **So most of the "~30% below V12 for 2.5 h" after fault 1 is a pod-4 slowdown of the restarted server.** The README
  puts it down to the longer restarted contexts, but at matched load those contexts are not the main cause.
  - The 1.16M thrown-away tokens (the ~40 min figure) stand.
  - The 2.5 h deficit after the restart should not be counted as a general restart cost of the solver design (R9).

## Method and data
- **Engine lines:** "Engine 000" 10 s lines, parsed from each server.log:
  - V1: `replicas/v1_qual/pod4_v1_qual_bundle.tgz` (start 0 and r1).
  - V12: `replicas/manh_v12/logs_bundle.tgz`.
  - control: `../../phase_h/sft_025ep_v12/data/gpumem.log`.
  - V1 on pod 2: `../../phase_h/runs/r1_rerun_bec71fb1/dsv41_logs/{server,r1/server}/server.log`.
  - Kaggle h is measured from each notebook's start.
- **Matched-load ratio:** the mean gen tok/s per (running // 4, KV% // 5) cell, over lines with prompt < 400 tok/s.
  Cells are weighted by min(n) where both runs have >= 3 lines. The bootstrap resamples lines; autocorrelation makes the
  CIs optimistic.
- **Tokens:** cumulative gen tok/s x line interval.
- **Tasks and score:** the acc_series method on each attempts.jsonl against `evalx2t_s2026_solutions.json`.
- **GPU clocks:** from `mon.jsonl` (V1) and `samp.jsonl` (V12, control).
- **Solver diff:** the pinned copies extracted from each notebook's cell 0b; the harness 882c817f, train_check and
  auto_repair are byte-identical.
