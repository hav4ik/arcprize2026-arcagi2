# E055: why did V5 score 58.19 (projected ~66)? Full 12 h Kaggle replicas on L4 pods

Question (user, 2026-10-26 ~21:30 UTC): "We should find out why the result is so low. 2-3 pts below is expected,
but not 8." v3 was predicted within 2 pts (52.5 vs 50.97); V5 (per pair, x36, no gate) missed by ~8.

Workload: the public test file is NOT a proxy (all 240 of its tasks are ARC-AGI-2 training tasks; 1,310 grid cells
per task vs 2,607 for eval; 7% multi-input vs 41%). The private set has ~335 pairs over 240 tasks (1.40 test inputs
per task, like eval's 1.43). So the replicas run **eval x2**: the 120 eval tasks twice under new ids
(`eval_x2_challenges.json` / `_solutions.json`, `eval_x2_idmap.json` maps copy id -> original): 240 tasks / 344 pairs,
with solutions, so each run is scored.

Runs (both started 2026-10-26 ~21:43 UTC, Kaggle default deadlines: stop-new 10.83 h, force 11.17 h, hard stop
11.5 h, kill switch 11.75 h; Runpod 4xL4 at util 0.938 = Kaggle's 1.554M-token KV pool):
- `v5rep_x2` on pod wd44jicex44qyc: **exact V5 replica** — V5's Kaggle kit (`kaggle/dsv41_tools/kaggle_kit_v5`: solver
  62144e4a, harness 1ad5ee24, train_check 44c4d210), per pair, x36, no gate, Humming, cudagraph cap 64 (`launch_v5rep.sh`).
- `cand_x2` on pod hegn9880s1tz6m: **next candidate** — multi-test + budget 0.5 + context limit, x44 + KV gate 0.7,
  retention interval 512, new code (solver c760fae0, harness 882c817f) + auto-repair (`launch_candx2.sh`).

Known optimism of a pod vs Kaggle: 72 W vs 50 W (prefill ~25-30% faster on the pod), no ~6 min install/staging
before the server (ARC_SKIP_INSTALL=1, local model). Questions each run answers: score on a hidden-sized workload,
KV over time (thrash in hours 5-11?), attempts finished / cut by the deadline, restarts / OOM, tok/s by hour.
If `v5rep_x2` scores ~58-61: the failure mode is reproduced and its logs show the cause. If it scores ~66+: the
gap is Kaggle-side (hardware, a harder private set) rather than the configuration.

## Late-phase replay (user idea, 2026-10-26 ~22:30 UTC): preload the KV to test hour-8 pressure in minutes
`late_replay.py` = E051's `tool_replay_kv.py` (itself E042's `tool_replay.py`: slots start mid-attempt with the history
sent cold, i.e. prefilled during the warm-up, then steady-state tool turns with random token ids, exact completion
lengths and recorded tool-result lengths) + `--tail-frac F` (draw every attempt from the heaviest F of the shapes, like
the late hours of a smallest-first 12 h run) and `--min-start-ctx N` (each slot's cold start is at a history >= N).
Shapes: `shapes_e128t_tools.json` (334 E128 attempts, `make_shapes.py` from E043's `tool_shapes_e128t...json`; final
context median 46k, p70 67k, p90 95k, max 120k). "Kaggle hour 8": `--tail-frac 0.3 --scale 1.25 --min-start-ctx
30000 --tool-delay 20 --conc 36|44|48`, pool `--num-gpu-blocks-override 21490`, offload off vs on. Limits: fixed
concurrency (no gate), random ids, no accuracy; complements, does not replace, the 12 h pipeline runs.

## Replica toolkit (`toolkit/`, 2026-10-27 ~01:10 UTC) and what it changed
- The leaderboard metric is task-weighted (58.19 = 139.67/240, 50.97 = 122.33/240 exactly): use task-weighted scores.
- Pods are ~10% faster than Kaggle end to end (one twin pair: Kaggle bench V1 vs pod `mt36_b05_gate60`: prefill 1.45x
  slower, decode 6% slower on Kaggle); on Kaggle's clock the force deadline falls at pod time ~8.9-9.8 h, so a pod gets
  ~1.4-2.3 h extra. `replica_report.py` applies this Kaggle-clock correction; `replica_run.sh` also shifts deadlines
  600 s earlier. The live `v5rep_x2` / `candB_x2` (identical copies sharing a ~7.1k first prompt, unshifted
  deadlines) are optimistic in absolute terms but compare fairly with each other; read their Kaggle-clock scores.
- New workload `toolkit/workloads/evalx2t_s2026_*`: copies under a seeded D8 + colour-permutation transform (no shared
  prefixes) for future replicas.
- Solver bug found (fix-list S15): engine deaths used up job tries; fixed in solver a435ddf4.

## Shared: B32 replica on Hugging Face (2026-10-27)
Private dataset `livobereshka/arc-agi2-dsv41-e128-candb32-replica`: the complete `candBf_x2t` run (attempts, 346 traces,
metrics, solver/server logs, submission), the `evalx2t_s2026` workload with solutions and idmap, the replica report, and
the pinned code. Secret scan (incl. extracted traces) 0 hits before upload.
