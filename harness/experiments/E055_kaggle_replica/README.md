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

## Provenance audit: are the V5 / V6 replicas genuine, or fitted to the leaderboard? (2026-10-27 18:35 UTC)
Checked because a replica that "matches" the LB is only evidence if nothing in it was tuned to the LB.

| item | how it was set | LB-independent? |
|---|---|---|
| Code | each replica ran the exact frozen notebook code, hash-checked: V5 = Kaggle V5 kit (`/root/kit_v5`, V5_GAP section 1); V6 = Kaggle V5 kit + V6's pinned solver 4b344011 / harness a8a5b665, kit train_check 44c4d210, kit overlay 377d83e0 (`launch_v6rep_x2t.sh`) | yes |
| Settings | copied from the submitted notebooks: V5 = 36 workers, per pair, no gate; V6 = 36 workers, multi-test 0.5, context limit 163840, gate 0.6/16 (both replicas' `env.txt`) | yes |
| KV pool | GPU util 0.938 so the pool equals Kaggle's 1,554,183 tokens (from V5/V6 Kaggle server logs) | yes (hardware match) |
| Workload | 120 public eval tasks ×2 (V5: identical copies, built before the toolkit; V6 and B32: `evalx2t_s2026`, seeded transforms) | yes |
| Kaggle-clock factors (prefill ×1.45, decode ×0.94, server up at +0.24 h) | measured on one throughput twin pair (Kaggle bench V1 vs pod `mt36_b05_gate60`, same V6 settings, 3 h, token counts), `calibrate_twin.py` | yes (no score enters; in-sample on that pair) |
| Metric | task-weighted, identified because 58.19 and 50.97 are exact multiples of 1/(3·240) | the metric's definition only |
| Scoring code | `toolkit/replica_report.py`: the only LB numbers in the toolkit are in that metric explanation (grep) | yes |
| Independent re-score | scoring each run's own `submission.json` against the solutions with a 10-line script: V5 57.57, B32 79.10, identical to the report | yes |
| Timing | V5's replica launched 21:43 UTC 10-26, ~13 min after V5's LB (58.19) was known, but with nothing chosen after it; B32's replica launched 04:08 UTC 10-27, before V6's LB was known; V6's replica launched 10:03 UTC 10-27, after V6's LB | — |

Where judgment still enters (disclosed, not hidden):
- **The Kaggle clock decides where V5 and V6 are cut.** V5 was stopped at pod hour 9.99, after the central clock's force point (9.88); its
  raw and central scores are both 57.57 (identical at 9.87 h and 9.99 h, so the stop time was not picked). Under the
  pessimistic clock (prefill ×1.6) V5 scores 55.9; under the optimistic clock (×1.3, decode ×1.0) the force point is at pod
  10.67 h, which this run never reached, so V5's optimistic score is unmeasured and would be a few points higher. So V5's
  match within 0.6 depends on the central clock, which came from the throughput twin, not from the LB.
- **B32 depends far less on the clock than V5 or V6:** it finishes every first attempt by pod 8.63 h, so slowdowns from ×1.0 to
  ×1.3 move it only 79.1 → ~77.0. V5 and V6 are deadline-bound and much more clock-sensitive.
- **V6 projections changed** (52-56 at 6.1 h, then 61-66 at 8.1 h) because the data changed; no parameter was retuned.

**Pre-registered now, before V6's measured value (pod 9.78 h, ~19:50 UTC):**
1. V6's calibration number = `replica_report.py`'s central Kaggle-clock score of `v6rep_x2t`, with no re-choice of clock or stop time.
2. B32's estimate = B32's central Kaggle-clock score (77.43) + the mean (LB − replica) gap of V5 (+0.62) and V6, with the
   ±range from two draws (replica sd ~3.1-4.2 per run, LB draw sd ~4). Nothing else is adjusted.

**Result (2026-10-27 20:00 UTC, rule applied as written):**
- V6's calibration number: **64.93** (`toolkit/reports/v6rep_x2t_kclock.md`; the run reached the central clock's force
  point at pod 9.77 h; optimistic 65.35, pessimistic 63.26). LB 61.94, so the replica is **2.99 points optimistic** for V6.
- Gaps (LB − replica): V5 +0.62, V6 −2.99; mean **−1.19**.
- **B32's estimate: 77.43 − 1.19 = 76.2.** Spread: replica sd ~3.6 + LB draw sd ~4 → ~±5.4 (1 sd), 80% range ~69-83.
  The two calibration gaps differ by 3.6 points, about one replica sd, so the mean gap itself is uncertain by ~±2.
- For comparison, the judgment-based "realistic ~74 (67-80)" given before this point stays inside the range; the
  pre-registered number is the one to quote.

**Final runs (2026-10-28 00:45 UTC; Runpod stopped by the user for budget):**
- V6 replica `v6rep_x2t` finished at 11.01 h: raw 69.24, central Kaggle-clock score **64.93** (unchanged from the
  calibration point; optimistic 68.68, pessimistic 63.26), `toolkit/reports/v6rep_x2t_final.md`.
- B32 draw 2 `candBf2_t2` (workload `evalt2_s2027`) was stopped by SIGTERM at pod 4.94 h of ~11 h (final write done,
  submission valid): 141 of 240 tasks with a first attempt, score so far 39.17, per-attempt accuracy of finished
  attempts 65.2%, 5.63M output tokens in finished attempts (draw 1 at the same pod time: 158 tasks, 5.72M).
  Too early for a second full-run estimate; B32's estimate stays 76.2 from draw 1. `toolkit/reports/candBf2_t2_partial.md`.

**B32 leaderboard (2026-10-28, user): 68.89**, against the pre-registered 76.2 (80% range ~69-83). Kaggle-clock replica 77.43, so LB - replica = -8.5 (V5 +0.6, V6 -3.0): the replicas get more optimistic as the config gets heavier.
- In the replica, 68.89 matches a ~1.55x Kaggle slowdown (x1.5 -> ~71, x1.6 -> ~67).
- Candidate causes: a server hang (#58947 deadlock, no watchdog in B32); Kaggle slowing the offload/host-restore traffic more than pods do; a harder hidden set. Scoring logs are not visible.
