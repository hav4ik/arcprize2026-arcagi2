# Why V5 looks ~7 pp worse per attempt than the E128 reference (2026-10-26/27, CPU-only analysis)

> **Verdict (2026-10-27 ~04:30 UTC, after the overnight replicas; details in section 6).** V5's 58.19 is what V5's
> configuration scores; nothing in V5 broke. The 8-point miss against the old projection (~66) had two causes:
> (1) the simulator's per-pair accuracy pool (E043 `w10_u2`, 2 samples per pair) was ~4-6 pp optimistic, and
> (2) the remaining 2-3 points are within the ±4-6 spread of one 240-task draw. With honest pools V5 projects
> 60.9 ± 6.0 (LB 58.19) and v3 47.6 ± 4.8 (LB 50.97): both inside 1 sd, with opposite signs, so no systematic bias is
> left. The live 12 h V5 replica (`v5rep_x2`, L4, Kaggle's KV pool) confirms it: at 6.6 h, 0 thrash windows (max KV
> 0.72), 0 preemptions, 0 restarts, a flat ~328 gen tok/s, and 62.6% per-pair accuracy, the level that gives ~58 after
> coverage. **For V6:** honest projection 69.1 ± 4.7 (task-weighted). On the same 132 pairs, one-rollout-per-task
> plus the new code (the first candidate-B replica) scored 75.8% against V5's 67.4% (17 pairs only B solved, 6 only
> V5; sign test p ≈ 0.035). V6 has the multi-test half of that gain but not the new code (~+2.4, E052), so expect
> roughly 65-70.

Question: Kaggle submit V5 (per pair, x36, no gate) gives 79/118 = 66.9% per attempt on the pod (`pods/runs/v5_w36`),
against 73.7% for the E128 reference on the same pairs (−6.8 pp). The V6 and next-candidate configs show no gap. Does
this explain the 58.19 leaderboard score (projected ~66)?

**Answer: V5 has no per-attempt accuracy gap. The gap is in the reference.** No harness bug, flag or server setting
costs V5 a single attempt. The reference, E043's `e128t_w10_u2_gptq_eng64_sub80` (2 samples per pair), scores higher
than every other E128 per-pair measurement on the same pairs. Against per-pair E128 runs with the Kaggle flags, V5 is
at **−0.5 pp (both V5 runs, 161 attempts, null sd 2.4, z −0.2)**.

V6 shows no gap for two reasons:
- it is scored against multi-test references;
- one rollout per task really is better on multi-input tasks: per-pair attempts there score 65%, multi rollouts 72–77%.

The simulator's per-pair pool is the same optimistic w10_u2 file. With the accuracy corrected, V5 projects to
**~63 (Kaggle-flag accuracy) or ~60 (V5's own measured accuracy)** under the old KV/thrash assumptions, not 66.5.
That closes about half to three quarters of the miss to 58.19. The rest is within the simulator's spread over
240-task draws (sd 4–6 points).

Scripts, derived data, diffs and simulator outputs are in `v5gap/`. The live replica was analysed from a snapshot of
47 attempts taken at +1.75 h (`v5gap/v5rep_x2_attempts_snapshot_47.jsonl`). Its traces were copied read-only from the
pod.

## 1. What differs: V5 kit vs pod kit vs V6, single-test path

Kits. Each file was checked by sha256.

| kit | solver | harness | train_check | used by |
|---|---|---|---|---|
| Kaggle V5 (`kaggle_kit_v5/dsv41_kit.tgz` f24e624b) | 62144e4a | 1ad5ee24 (= snapshot commit f26a001a, 2026-10-24 21:08 UTC) | 44c4d210 | Kaggle V5; `v5rep_x2` (`/root/kit_v5` on the pod, hashes checked there) |
| podkit (`/root/podkit/dsv41_kit.tgz` a9706290 = `kaggle/dsv41_tools/out/kit_ds`) | 4a3ec371 | c1a05425 | 8fe36175 | `v5_w36` (`ARC_KIT=/root/kit`) |
| V6 (`pinned_mtf_v6/`) | 4b344011 | a8a5b665 | (kit's 44c4d210) | V6 and the V6-code pod and 4090 runs |

**podkit vs V5 kit:** identical on the per-pair path. Every harness difference is gated by `n is not None` /
`--multi-test` (`v5gap/harness_v5_podkit.diff`). The solver differences are `ARC_MULTI_TEST`, the KV gate
(off by default) and `ARC_TASK_IDS`. The train_check differences are opt-in arguments the solver never passes. So
`v5_w36` measured V5's behaviour.

**V5 (1ad5ee24) vs V6 (a8a5b665), single-test path** (`v5gap/harness_v5_v6.diff`):
1. `submit_answer` reaches built-ins through `import builtins as _b`, so a model-defined `eval`, `compile`, `exec`,
   `globals` or `hasattr` no longer breaks it. V5 is exposed to this; it happened 0 times in 170 V5 traces.
2. `submit_answer` gets `_plain` nested conversion on the single-test path, so a list of numpy rows is accepted. V5
   is exposed to this; see section 3 for the one occurrence.
3. `check_answer_on_train` gets the built-ins fix. It only runs with `--verify-submit`, which no run uses.
4. `--context-limit` / `ContextFull`, and `"context"` added to the session-fallback stops. V6 passes 163840; V5 has no
   guard. No V5 attempt came near the limit (0 errors).
5. Changes only in V6 that could hurt it: A5. With `--multi-test`, single-test tasks are told `test_inputs` exists, but
   their sessions do not define it.
6. Everything else is multi-only.

**Solver:** the `HARNESS` list and `EXTRA_BODY` are identical in V5, podkit and V6.
- Flags: `--mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens
  64000 --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal
  --turns-note 5 --train-check --rescue-code --token-exact --fallback-solve`, max turns 40.
- Request body: `reasoning_effort` 50, T 1.0, top_p 1.0, cache_salt.
- V6 adds only env-driven flags: `--multi-test`, `--multi-budget 0.5`, `--context-limit 163840`, KV gate 0.6/16.

**Server:** `server_cmd()` is unchanged between V5 and V6. The `server_cmd.json` of `v5_w36` and `mt36_b05_gate60`
(same pod type) are identical in cmd and env. E050's differs only in util (0.92 vs 0.938).

**How the reference runs differ:** `w10_u2` ran on the E043 8x RTX PRO 6000 box with TP8 and expert parallel (E043
`serve.sh`: max-model-len 131072, batched tokens 8192, util 0.90, no Humming dense, no `DSV41_INDEXER_Q_PREDECODE`).
It used the Docker sandbox and the chat path (no `--token-exact`), and it had no `--fallback-solve`. It did use
`--rescue-code`, with 0 rescues. `pp_sub80_tx` (per pair, Kaggle flags) and run 7 ran on E046's 4x RTX PRO 6000
boxes with `--token-exact --fallback-solve --rescue-code`.

## 2. Accuracy, matched by test-input content

Per-attempt accuracy of the references against each other, on the same eval pairs (`v5gap/refcompare.py`):

| pairs | E043 w10_u2 (chat path, 2/pair) | Kaggle-flag per pair | multi rollouts |
|---|---|---|---|
| single-test tasks (71) | **76.8%** | run 7 74.6%, V6-code L4/4090 runs 72.8% (353 att), new-code E052 74.2% | (same path) |
| multi-test tasks (92) | **69.0%** | pp_sub80_tx 65.3% (341 att) | run 7 74.5%, V6-code 73.1%, new-code 76.8% |

V5 runs against each reference (paired by pair; the null sd treats V5 as a draw at the reference's rate):

| run (attempts) | vs w10_u2 | vs Kaggle-flag per-pair ref* | single-test: vs w10 / vs Kaggle-flag | multi-test tasks, per pair: vs w10 / vs Kaggle-flag |
|---|---|---|---|---|
| `v5_w36` (118) | 66.9 vs 73.7, **−6.8** (z −2.8) | 67.5 vs 69.6, **−2.0** (z −0.7) | 71.7 vs 75.5 / vs 72.2 (−0.5) | 63.1 vs 72.3 / 63.9 vs 67.3 (−3.4) |
| `v5rep_x2` snapshot (47) | 68.1 vs 76.6, −8.5 | 68.1 vs 65.0, **+3.1** | 68.8 vs 78.1 / vs 73.2 | 67.7 vs 75.8 / vs 60.8 (+7.0) |
| both (165 / 161) | 67.3 vs 74.5, −7.3 | 67.7 vs 68.2, **−0.5** (z −0.2) | 71.0 vs 76.1 / vs 72.4 (−1.4) | 64.6 vs 73.4 / 65.2 vs 65.1 (+0.1) |

\* The Kaggle-flag per-pair reference uses, for single-test pairs, run 7 plus the V6-code runs (all on the same
single-test code path as V5). For multi-test-task pairs it uses `pp_sub80_tx`.

- On the pairs V5 finished, w10_u2 is 6.3 pp above the Kaggle-flag reference (74.5 vs 68.2). That accounts for
  nearly the whole "gap". On the full eval set it is +2–4 pp (about 1–1.5 sd). Its two samples per pair are lucky on
  several pairs that V5 missed:
  - 4c3d4a41.0: w10 2/2 vs 0/4 for the Kaggle-flag runs;
  - 4ba524c6.0: 2/2 vs 0/4;
  - 3a25b0d8.1: 2/2 vs 1/4;
  - 4e34c42c.0: 2/2 vs 1/4;
  - d35bdbdc.2: 2/2 vs 1/3.
- Tokens, turns and train-pass rate match the references on the same pairs. For example, single-test pairs in
  `v5_w36`: 28.5k tokens, 15.0 turns, 75% train-pass, against 28.9k, 15.4 and 80% for w10.
- V5's no-answer rate is 5.9%, against 8.1% for w10 on the same pairs.
- The server and harness health counters are the same as in the reference runs:
  - tool errors per call 0.15 (refs 0.13–0.15);
  - 0 rescues and 0 cuts;
  - token-exact breaks 0.28 per turn (refs 0.27–0.30) and 3 resyncs;
  - 0 wall-clock notes;
  - tool timeouts 14 of 2,410 calls (refs 0.25–0.34%).
- The one notable difference: V5 has more wrong-but-train-passing answers (P(correct | train_pass) 72–74% vs ~81%
  on the same single-test pairs). V5 offsets this with more train passes. It is within noise (about 1.5 sd) and is
  not a harness effect: train_check is the same code, and the answers are the model's own.

## 3. Failure classes: every V5 attempt that is wrong or has no answer (traces classified, `v5gap/classify.py`)

`ref-weighted loss` is the sum of the reference's success rate over V5's wrong attempts in that class. `ref right`
counts wrong attempts whose pair the reference solved in at least half of its attempts.

| class | `v5_w36`: n / ref right (w10) / loss vs w10 / loss vs Kaggle-flag ref | `v5rep_x2`: n / ref right (w10) / loss vs w10 / loss vs Kaggle-flag ref |
|---|---|---|
| submit_answer rejected → lost | 0 / 0 / 0 pp / 0 pp | 0 / 0 / 0 / 0 |
| eval/builtins shadowing | 0 (no `def`/assignment of eval, compile, exec, globals, hasattr, print, json, … in any of 170 traces) | 0 |
| answer lost on a cut / error (A2/A4) | 0 (0 cuts, 0 errors, 0 parse/lenient answers) | 0 |
| budget or turn cap, no answer | 7 (+2 without a reference) / 0 / **0.0 pp** / 0.4 pp | 0 |
| wrong, submitted, train_pass True | 22 / 16 / **8.9 pp** / 6.6 pp | 12 / 8 / 10.6 / 3.4 |
| wrong, submitted, train_pass False | 9 / 5 / **3.8 pp** / 2.6 pp | 3 / 3 / 4.3 / 3.2 |
| wrong session answer at the turn cap (other) | 1 / 0 / 0 / 0 | 0 |
| V5 right where the reference is not (gain) | 79 correct: **+5.9 pp** / +7.5 pp | 32 correct: +6.4 / +9.7 |
| **net** | **−6.8 pp vs w10 / −2.0 pp vs Kaggle-flag** | −8.5 / +3.1 |

- **Every lost point is a model miss.** Of the 21 `v5_w36` attempts that are wrong where w10 is at least 50% right,
  all 21 submitted a wrong grid. Each was off by a few cells, the wrong shape, or a transposed shape (78332cb0: 5x23
  vs 23x5).
- Strong misses, where the pooled per-pair rate (w10 plus Kaggle-flag runs) is at least 80%, all go through train-verified programs that
  generalise wrongly or fit only part of train: 6e453dd6.0, 97d7923e.0, 9385bd28.0, 6ffbe589.0, 8e5c0c38.0 and
  20270e3b.1 (v5rep). In 97d7923e the model's `solve` reproduces all 3 training pairs and is 7/200 cells off on the
  test.
- **The no-answer attempts have no `answer` and no working `solve`.** The session fallbacks report
  `NameError: name 'answer' is not defined`. On the `solve` side: 7 attempts have no `solve`; 581f7754's raises
  `UnboundLocalError`; f560132c's is `solve(grid, W)` (`missing 1 required positional argument: 'W'`). None of the
  A1–A11 fixes would rescue them. w10 is 0/2 on all 7.
- **Rejections seen:** 7 attempts in both runs had a rejected `submit_answer`, and 6 of them ended correct:
  - 5× `NameError: name 'answer' is not defined` (the model submitted before setting it);
  - 1× `ValueError` inside the model's solver;
  - 1× `json.dumps` failure (v5rep 9ed3044f.1: `solve` returned a tuple). `_plain` would not have accepted that
    tuple either; the model fixed it and was right.

  The one rejected attempt that ended wrong (v5rep dbff022c.0: the reference is 0/2, and every per-pair source 0/7)
  submitted again after the rejection, so the rejection cost nothing.
- **Harness bugs cost 0 attempts → 0 pp, in both runs.** The numpy-row (`_plain`), builtins and A2/A4/A5/A9 classes
  do not occur. Under any reference, the whole loss sits in "wrong-but-submitted".

## 4. Was the reference a fixed-harness run? No: it is older than V5's harness

From file times and the snapshot repo history (`arcprize2026-arcagi2`, `havu/evals/run_deepseek_tools.py`):

- **`e128t_w10_u2_gptq_eng64_sub80`:** finished 2026-10-24 ~13:07 UTC (committed 13:13 UTC). That is ~6 h before
  `--token-exact` existed (commit baeed56b, 19:25 UTC) and ~8 h before V5's harness 1ad5ee24 (f26a001a, 21:08 UTC).
  - It ran a ≈80d14a5e-era harness (commit 2345e933): no `import builtins as _b`, no `_plain`, no context guard, no
    fallback-solve, chat path.
  - So it has every V5 bug. The bug fixes cannot be the difference.
- **`pp_sub80_tx` and run 7 (`e128t_final_all`):** 2026-10-25 evening (UTC 10-26 05:5x–06:20). Their harness lies
  between 1ad5ee24 and 90ef574e (snapshot 2c1b443c, 10-26 07:51 UTC). Run 7 has the builtins fix and multi `_plain`
  but not single-test `_plain` (E046 README: 800d221b).
- **E052 runs:** all the A-fixes (13ebdb70 / 882c817f).

The "fixed" runs do not score higher on single-test pairs: run 7 74.6%, V6-code 72.8%, new-code 74.2%, against w10's
76.8% with no fixes. The fixes' measured gain (E052, ~+2.4 pp) is in multi-input tasks on the multi path (A1, A6, A9),
which V5 never takes.

## 5. Conclusion

- **Harness bugs:** none cost V5 anything (0 of 170 attempts).
- **Flags and settings:** V5's flags, request body and server match V6 exactly on the single-test path, and V5 ties
  V6-code runs there (71.0 vs 72.5 on the same 69 pairs).
- **What is left:**
  - **(a) Reference optimism.** w10_u2 is the high outlier among E128 per-pair measurements (+2–4 pp on the full set,
    +6 pp on V5's pairs). The cause is luck with 2 samples per pair, possibly plus a small chat-path vs token-exact
    or 8x-box effect that cannot be separated without a GPU rerun.
  - **(b) Normal noise.** Against the Kaggle-flag reference: `v5_w36` −2.0 ± 5.7 pp (95%), `v5rep` +3.1, pooled −0.5.
  - **(c) The real V5 vs V6 accuracy difference.** It comes from the config, not a bug: per-pair attempts on
    multi-input tasks (~45% of eval pairs) score ~65% against ~72–77% for multi rollouts.
- **Tooling note for E055:** `bench_accuracy.py load_bench` keys attempts by (pair key, attempt). On `eval_x2` the two
  copies of a task share pair keys, so the copies collapse (the mirror's 43 attempts read as 25). Key by
  (task, pair, attempt) for x2 runs; `v5gap/common.py` does this.

### Re-projection with the measured accuracy

The setup follows `goal75` (`multi_sim`, 240 tasks drawn from eval, V5 = per pair x36, KV 1.554M, s 1.04, 40
reps x 2–3 flip seeds). The per-pair pool's right answers are flipped to wrong so that pool accuracy equals:
- the Kaggle-flag per-pair accuracy (single x0.955, multi x0.946; pool 71.9 → 68.6%), or
- V5's own measured ratio to w10 (x0.903; pool 64.4%; 95% CI x0.85–x0.96).

| per-pair accuracy | no thrash (KV/ctx 0.72, as on pods) | 0.83 + thrash 0.88:0.6 (old assumption) | v3-consistent ratio* | 0.95 + thrash (v5calib) |
|---|---|---|---|---|
| w10_u2, as projected | 69.5 | 66.5 | — | 58.7 |
| Kaggle-flag per pair | 65.7 | **62.9** | 64.6 (0.78; v3 → 51.3) | 55.4 |
| V5-measured (x0.903) | 63.0 | **60.3** | 62.8 (0.72; v3 → 52.3) | 52.9 |
| V5-measured, CI ends x0.85 / x0.96 | 59.2 / 66.2 | 56.6 / 63.4 | | 49.6 / 55.9 |

\* The ratio at which v3 (x48, KV 1.44M) reproduces its LB 50.97 with the same corrected accuracy.

- **Does it land near 58.19?** Closer, but still 2–5 points above it.
  - With the corrected accuracy, V5 projects to 60–63 under the old assumptions, or 62.8–64.6 at a v3-consistent
    calibration.
  - The accuracy correction removes 3.6–6.2 of the ~8.3-point miss. The remaining 2–5 points are within the
    simulator's sd over 240-task draws (4.0–5.8 points).
  - Late-run thrash on the heavier private set remains a possible extra cause; the V5 replica will show it.
- **Calibration consequence:** the "KV per context token 0.95 fits both v3 and V5" calibration from findings
  section 11.6 no longer holds once accuracy is corrected. At 0.95, v3 projects 45–47 and V5 53–55, both too low.
  No single ratio fits both; v3 wants ~0.72–0.79, V5 ~0.86–0.90.
- **Side effect on V6:** grid.py uses the same w10_u2 pool for V6's single-test tasks. Correcting it moves V6's
  projection from 70.3 to 68.5 at 0.83 + thrash, and from 68.0 to 66.2 under the 0.95 recalibration (`v6proj.py`).
  Every per-pair or mixed projection built on w10_u2 is optimistic by ~3–4 pp on per-pair pairs. The simulator's
  per-pair pool (`multi_sim.per_pair_pool`, `deadline_sim.POOLS["E128"]`) should switch to the Kaggle-flag
  per-pair records:
  - single-test tasks: run 7 plus the V6-code runs;
  - multi-test tasks: `pp_sub80_tx`.

## 6. Live evidence (2026-10-27 overnight): the V5 replica and a paired comparison

**The replica.** `v5rep_x2` runs the exact Kaggle V5 kit (`/root/kit_v5`, hashes checked) on a Runpod 4xL4 pod at
util 0.938 (1,556,850 KV tokens = Kaggle's 1,554,183), on `eval_x2` (240 tasks / 344 pairs: every eval task twice),
from 2026-10-26 21:43 UTC with Kaggle's default deadlines. Polled every ~1.5-2 h (`poll_e055c.sh`):

| elapsed | attempts | per-pair correct | mean gen tok/s | max KV | thrash windows | preemptions | host RSS |
|---|---|---|---|---|---|---|---|
| 4.45 h | 130 | 64.6% | 328 | 0.668 | 0 | 0 | 114 GiB |
| 6.04 h | 176 | 64.2% | 328 | 0.716 | 0 | 0 | 115 GiB |
| 6.63 h | 195 | 62.6% (15 no answer) | 328 | 0.716 | 0 | 0 | 115 GiB |

- **No late-run collapse so far.** x36 ungated per-pair attempts stay below the KV level where thrash starts
  (~0.83-0.89), unlike the multi-test x56 configuration (below). The thrash hypothesis for V5's miss is not supported.
- **Accuracy is the driver.** 62-65% per pair on the answered pairs (smallest tasks first) matches the Kaggle-flag
  per-pair references (section 2), not w10_u2.
- **Caveats on absolute numbers** (E055 toolkit, `toolkit/README.md`): the leaderboard metric is task-weighted
  (58.19 = 139.67/240, 50.97 = 122.33/240 exactly); pods are ~10% faster than Kaggle end to end (prefill ~1.45x,
  decode ~1.06x; one twin pair), so a pod gets ~1.4-2.3 h of extra generation before Kaggle's force deadline; and
  `eval_x2`'s identical copies share a ~7.1k-token first prompt with their originals. `replica_report.py` corrects the
  first two (Kaggle-clock section); read the V5 replica's Kaggle-clock score, available once the pod passes
  Kaggle's force deadline on that clock (~pod hour 9.8, ~07:30 UTC), as the calibration point against 58.19.

**Paired comparison with the first candidate-B replica** (`candB_x2`, same pod type, same workload; next-candidate
code + auto-repair, multi-test, interval 512, KV offload 28 GiB, x56 + KV gate 0.85; stopped at 4.05 h after it found
the gate bug S17). On the 132 pairs both had answered by 03:45 UTC (attempt 0 only):

| | V5 replica | candidate B |
|---|---|---|
| per-pair correct | 67.4% | **75.8%** |
| solved by this run only | 6 | 17 |
| solved by both | 83 | 83 |

The +8.4 pp is the multi-test effect (one rollout per task on multi-input tasks) plus the new code (E052: ~+2.4), on
the same pairs and hardware. Sign test on the 23 discordant pairs: p ≈ 0.035.

**What this means for V6** (multi-test + gate 0.6, V6 code): it has the multi-test part of the gain but not the
new code. Honest projection 69.1 ± 4.7 (`goal75/honest2.py`); with the replica evidence, expect roughly 65-70. V6's
real score (pending) is the next calibration point; a V6 replica on L4 would pair it with these runs.

**Lessons recorded** (findings note 11.7-11.8): project from each configuration's measured accuracy on the same
hardware; always give the ±sd of a 240-task draw; validate late-run behaviour with a 12 h replica, not the simulator;
treat a measured accuracy gap as a signal, never as noise to explain away.
