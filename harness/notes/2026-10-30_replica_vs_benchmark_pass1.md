# 2026-10-30: replica first-attempt pass@1 vs the no-limit benchmark, paired on the same tasks

**Question:** is the first-attempt accuracy of the two running 12 h replicas (Manh V12 on pod 2, B32-M V3 on pod 3)
below a no-limit benchmark of the same model and harness? That would mean the Kaggle-like setup (deadlines, KV gate,
budget notes, offload, numerics, restarts) costs quality. Early numbers are biased by which tasks finish first, so the
comparison is paired on the same original tasks.

**Data as of 05:40 UTC:** V12 at Kaggle clock 3.77 h with 170/240 first attempts; B32-M V3 at 2.03 h with 63/240.

**B32-M V3's server died at 05:42 UTC (+2.20 h).**
- Before the death gen tok/s fell from ~475 to 148 over 5 min. The solver log reads "server died (code 0)".
- All 80 in-flight first attempts ended as error records (APIError) and were re-queued; restart 1 started at 05:43.
- The script counts error records as unfinished, so B32-M V3's numbers below are its 63 attempts from before the death.

## Verdict

**No gap shows yet in either replica.** The point estimates are about -2.5 pp, inside the noise.

| replica | first attempts | replica pass@1 | bench, same tasks | paired diff [95% CI] | tasks better / worse | fully correct diff |
|---|---|---|---|---|---|---|
| Manh V12 | 170 (87 orig. tasks) | 72.6 | 75.2 | **-2.5 [-8.3, +3.4]** | 24 / 29 | -4.4 [-11.0, +2.3] |
| Manh V12, complete queue prefix (no selection) | 99 | 68.7 | 71.5 | **-2.9 [-11.2, +5.7]** | | |
| B32-M V3 | 63 (38 orig. tasks) | 84.9 | 87.3 | **-2.4 [-9.0, +3.4]** | 4 / 5 | -3.2 [-10.3, +3.2] |

- **Re-read at 05:45 UTC.**
  - V12: 175 first attempts, -2.5 [-8.1, +3.2] (fully correct -4.6 [-11.0, +1.7]). Its complete queue prefix is now
    144 tasks: 68.1 vs 71.1, **-3.0 [-9.7, +3.6]**. On the same 144-task prefix the finished replicas score B32 +1.0,
    fork x80 -6.1, B32+R1 -6.1.
  - B32-M V3: unchanged (restarting).
  - The tables below are the 05:40 snapshot.
- ARC Prize task score: a task scores the fraction of its test inputs answered. The benchmark is E052 A and acc_manh3
  pooled (next section). The CI is a bootstrap over original tasks; a task and its transformed copy are resampled
  together.
- With E046 run 7 as the benchmark the gaps are smaller: V12 -1.0 [-6.4, +4.5] and B32-M V3 +0.0 [-9.0, +8.2]. Pooling
  all three benchmark runs gives -1.9 [-6.6, +2.8] and -1.2 [-7.4, +4.9].
- **Caution from the finished replicas:** on the same workload, harness and model, two of three finished x2t replicas
  ended significantly below this benchmark:
  - Manh's fork x80 (`manh_x2t_w80`): -7.2 [-12.5, -2.1];
  - B32 + R1 (`b32r1_x2t`): -6.4 [-12.3, -0.7];
  - B32 itself (`candBf_x2t`) tied: -1.5 [-6.8, +4.0].

  Their gaps were already there in the first 99 tasks of the queue (-7.9 and -4.4; B32 0.0). At this stage of their runs
  the finish-ordered numbers showed only -1.2 and -4.6. So V12's -2.5 is consistent with both no gap and a fork-sized
  gap of about -7. The final numbers of both replicas, and V12's growing queue prefix (144 tasks at 05:45), will decide.

## What is compared

**Replicas.** Both run the same B32-era harness and prompt settings:
- harness `run_deepseek_tools.py` 882c817f, pinned in cell 0b of both notebooks;
- lean2sub, 80k / 40k output budgets, effort 50 plus the "minimal" header, multi-test with budget 0.5, context limit
  163,840, auto-repair, rescue-code, token-exact, fallback-solve;
- temperature 1.0, top_p 1.0.

| | Manh V12 (pod 2) | B32-M V3 (pod 3) |
|---|---|---|
| notebook | `kaggle/dsv41_tools/out/manh_v12_363638159/` (B32-M + Manh's external server) | `kaggle/dsv41_tools/out/b32m_v3_363310806/` (B32-M as submitted) |
| model | old tool E128, Kaggle copy `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit` | same |
| solver | 11bfaaaa = B32's 77fe80a3 + S25/S26 (external server and its supervisor; the diff touches nothing else) | 77fe80a3 (B32's) |
| server | Manh's launcher, vLLM g8cf0c0664b (v11); no DCP; KV 1.61M; SimpleCPUOffload 10 GiB/rank; batch-sharded sampling, watermark 0.1, retention 128 | Manh's wheel g5e1bccd07c + cell 2r; DCP4, interleave 64; KV 3.20M; 32 GiB offload; util 0.945 |
| load | 64 workers; KV gate 0.85 holds about 30 running from hour 1 | 80 workers, 80 running, KV ~0.4 |

**Workload:** `evalx2t_s2026`, confirmed by `b32m_v3/data/workload.txt` and V12's README. It has 240 tasks and 344
pairs: the 120 eval originals, byte-identical to `data/arc-agi_evaluation_*`, plus one D8 + colour-permuted copy of each.
The idmap maps every copy to its original. The script checks that every copy's test inputs are its original's under
the idmap transform, in order (OK).

**Benchmarks:** same model, no deadline, no binding KV.

| benchmark | file | code | stack / box | attempts | pass@1 |
|---|---|---|---|---|---|
| **E052 A `new_ar`** | `experiments/E052_next_code_gate/runs/new_ar/out/dsv41_logs/attempts.jsonl` | harness 13ebdb70 + auto-repair, solver 24aaa376: B32's code minus off-by-default levers and later solver fixes | B32 stack, 24 GB 4090 box, x36, gate 0.6 (KV peak 0.58) | 1 per task, Kaggle eval json | 76.5 |
| **acc_manh3** | `experiments/E062_manh_vllm_fork/acc48/runs/acc_manh3/out/dsv41_logs/attempts.jsonl` | **B32's kit exactly** (solver 77fe80a3, harness 882c817f, auto-repair) | Manh's fork b9ad670a6 with B32-M's flags (DCP4, interleave 64, embed UVA, 32 GiB offload), 48 GB 4090 box, KV 23.6M (peak 0.055), x56 | 1 per task, Kaggle eval json | 77.1 |
| E046 run 7 | `evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all.jsonl` | persistent-v6 before the E052 fix list, no auto-repair | 8x box | 2 per task, GitHub eval set (6 tasks with different prompts are left out, 114 used) | 74.0 on its 114 |

- **Primary: E052 A + acc_manh3, pooled to 2 attempts per task (76.8).**
  - acc_manh3 is the closest single match for both replicas: B32's exact kit on Manh's fork, with B32-M's server flags.
  - E052 A is the same code on B32's stack.
  - Pooling halves the benchmark's own draw noise; the two disagree by up to 3 pp per replica.
- Run 7 is the "73.7 / 84.6" number, but it predates the fix list and auto-repair, which E052 credits at about +2.4 pp.
  It sits 2.8 pp below the primary on the same tasks, so it is a sensitivity check only.
- Not used:
  - candidate 1 (new E128, 75.1): a different model from both replicas;
  - E056: Kaggle smokes of 3 tasks only;
  - `acc_b32` / `acc_r1` (about 68%): the known 48 GB box anomaly for B32's stack.

## Method

`experiments/E062_manh_vllm_fork/per_layer/scripts/replica_vs_bench.py`.

- **Replica side:** the last record of each attempt-0 job. Correctness is an exact grid match against the x2t solutions.
  Multi-test records carry one grid per test input. Every finished first attempt is mapped to its original through the
  idmap.
- **Benchmark side:** each attempt's pairs are matched to the original task by test-input content. Per-task pass@1 is
  the mean over its attempts, using only the pairs both sides have.
- **Reported:** n, replica and benchmark pass@1 on the same tasks, the mean paired difference with a cluster bootstrap
  over original tasks (10,000 reps), and tasks better / worse. Each figure is given as the ARC score and as "fully
  correct".
- **Pod report counts differ.** V12's pod reports count correct with `phase_h/scripts/first_correct.py`, which checks
  only the first test input of multi-test tasks: 97/138 = 70.3% at report 3. At that point the ARC score was 72.5%
  and fully correct 68.8%. B32-M V3's `first_correct` is fully correct.
- **Two selection effects.** Pairing removes the task-level one (small, easy tasks start and finish first). It does
  not remove the attempt-level one: on any task, a short attempt finishes early and is more often right. Controls:
  1. **Complete queue prefix.** The solver starts first attempts in a fixed order (prompt size, smallest first),
     whatever their outcome. A prefix of that order where every first attempt has finished is therefore free of
     attempt-level selection. This is the clean intermediate number.
  2. Progress-aligned paired differences of the three finished x2t replicas and of two no-limit runs used as
     pseudo-replicas.
  3. A length-censored benchmark: only benchmark attempts no longer than the replica attempt could have run by the cut,
     at its own token rate. It turned out weak: it never explained the finished replicas' drift.

## Results in detail

### Manh V12 (170 first attempts, 3.77 h; benchmark E052 A + acc_manh3)

| subset | n | replica | bench | diff [95% CI] |
|---|---|---|---|---|
| all finished | 170 | 72.6 | 75.2 | -2.5 [-8.3, +3.4] |
| originals only (identical prompts) | 85 | 73.7 | 74.9 | -1.2 [-7.5, +5.1] |
| transformed copies only | 85 | 71.6 | 75.5 | -3.9 [-11.5, +3.3] |
| complete queue prefix 0-98 | 99 | 68.7 | 71.5 | -2.9 [-11.2, +5.7] |
| vs acc_manh3 only / E052 A only | 170 | 72.6 | | -3.8 / -1.3 |
| vs run 7 (114-task benchmark) | 160 | 74.7 | 75.7 | -1.0 [-6.4, +4.5] |

- Copies vs originals is not a consistent effect: in the finished replicas the sign flips (B32: originals -2.4,
  copies -0.6; R1: -8.8, -4.0; fork: -6.0, -8.5).

Time profile, by the hour the first attempt finished:

| finished in hour | n | replica | bench, same tasks | diff | cum. n | cum. diff [95% CI] |
|---|---|---|---|---|---|---|
| 0-1 | 50 | 87.0 | 88.0 | -1.0 | 50 | -1.0 [-10.2, +8.5] |
| 1-2 | 40 | 70.0 | 64.2 | +5.8 | 90 | +2.0 [-5.4, +10.3] |
| 2-3 | 44 | 58.0 | 70.8 | -12.9 | 134 | -2.9 [-9.6, +4.0] |
| 3-4 (partial) | 36 | 73.6 | 75.0 | -1.4 | 170 | -2.5 [-8.3, +3.4] |

- **The fall from 87% to 58% is mostly selection.** The benchmark falls with it (88 → 64-71) on the same tasks.
- The -12.9 in hour 2-3 is within what finished replicas show for single hours from attempt-level selection plus noise
  (B32 had -15.4 at 5-6 h and -22.2 at 8-9 h). It is not tied to the CPU-restore phase: hour 1-2, with similar host prefix
  hits (86-94% vs 95-96% in hour 2-3), came out +5.8.

### B32-M V3 (63 first attempts, 2.03 h)

| subset | n | replica | bench | diff [95% CI] |
|---|---|---|---|---|
| all finished | 63 | 84.9 | 87.3 | -2.4 [-9.0, +3.4] |
| originals / copies | 33 / 30 | 86.4 / 83.3 | 87.9 / 86.7 | -1.5 / -3.3 |
| length-censored benchmark | 58 | 85.3 | 90.5 | -5.2 [-11.9, +0.0] |
| vs acc_manh3 only / E052 A only | 63 | | | -5.6 / +0.8 |
| vs run 7 | 63 | 84.9 | 84.9 | +0.0 [-9.0, +8.2] |
| first 99 of the queue | 55 of 99 finished | 82.7 | 85.5 | -2.7 [-10.2, +4.4] (incomplete) |

- Hours: 0-1 h has 19 attempts (78.9 vs 81.6, -2.6); 1-2 h has 43 (87.2 vs 89.5, -2.3).
- There is no complete queue prefix yet: all 80 first attempts started at 0.24 h, and 37 of the first 80 had not finished when the server died.
- Too early for a clean number. The selection-corrected estimates (length-censored -5.2, acc_manh3 alone -5.6) lean
  negative, but the CIs reach 0.

### Selection calibration: finished x2t replicas and no-limit pseudo-replicas (primary benchmark)

| run | paired diff after 63 finished | after 170 finished | complete prefix 0-98 | final (240) |
|---|---|---|---|---|
| `candBf_x2t` (B32, x56, KV 1.55M) | -0.8 | +0.9 | +0.0 [-7.2, +7.7] | -1.5 [-6.8, +4.0] |
| `manh_x2t_w80` (Manh's fork, x80, KV 3.06M) | -2.4 | -1.2 | -7.9 [-16.8, +0.7] | **-7.2 [-12.5, -2.1]** |
| `b32r1_x2t` (B32 + R1, x56, KV 2.49M) | -2.9 | -4.6 | -4.4 [-12.2, +3.9] | **-6.4 [-12.3, -0.7]** |
| no-limit acc_manh3 as pseudo-replica (vs E052 A + run 7) | +6.4 | | | +2.8 [-3.3, +8.8] |
| no-limit E052 A as pseudo-replica (vs acc_manh3 + run 7) | -0.1 | | | +2.2 [-4.8, +9.0] |
| **Manh V12 now** | +0.0 | **-2.5** | **-2.9 [-11.2, +5.7]** | |
| **B32-M V3 now** | **-2.4** | | n/a | |

- **The finish-ordered difference drifts down over the run.** From this point to the end the finished replicas moved
  -0.7 to -6.0 pp, and the pseudo-replicas -3.6 and +2.3.
- **The queue prefix does not drift.** On the first 99 tasks the finished replicas already sit at their final level.
  So their final gaps are not a late-run (KV-pressure or deadline) effect; the early finish-ordered numbers simply hid
  them.
- None of those replicas cut a first attempt at the deadline.
- **Manh's fork at x80 on L4 lost 7 pp first-attempt accuracy against the same fork on the 48 GB box.** This is not
  noted elsewhere yet. It is only partly visible in E062's earlier "-3.2 [-9.6, +2.5] (124 tasks, cross-pod)".
  - Against run 7 alone the three finals are +2.9 / -2.8 / -2.3, so part of it depends on the benchmark draw.
  - Two of three replicas land at -6 to -7 against the matched benchmark.
- **Both current replicas run Manh's vLLM**, so the fork's L4 gap is the relevant prior for them.

### Where the replica loses: V12, 29 of 170 tasks below the benchmark (summed loss 16.7 tasks vs 12.3 won)

- **Stops:** 24 submitted a wrong answer, 20 of them with train_pass; 2 output_budget, 2 max_turns, 1 context.
  - 0 deadline or loop cuts and 0 errored attempts.
  - 0 restarts, OOMs, APITimeoutError or HTTP 500 (pod reports).
- **Length:** 58.8k output tokens and 23.0 turns per attempt, vs the benchmark's 49.5k on the same tasks. Tool errors
  2.90 vs 2.55 per attempt. The benchmark got 42 of those 58 attempts right.
- **No mechanical signature.** All paired tasks vs the benchmark's attempts on them:

  | | replica | benchmark |
  |---|---|---|
  | output tokens (ratio) | 1.00 | |
  | turns | 17.2 | 17.3 |
  | train_pass | 77% | 75% |
  | tool errors / attempt | 1.78 | 1.78 |
  | no-answer stops | 13/170 (7.6%) | 15/340 (4.4%) |

  - Token-exact retokenization breaks: 27% vs 30% in acc_manh3. Resyncs 4, rerenders 0, auto-repair 149 applied.
- **The losses look like ordinary sampling misses**: a wrong rule that fits the training pairs, in somewhat longer
  attempts. They are not truncation, budget or serving failures.
- **Repeat offenders:**
  - 7b5033c1 and its copy fail in both replicas, in E052 A and in both run 7 attempts; only acc_manh3 solved it.
  - 6ffbe589 hits max_turns everywhere.
- **B32-M V3:** 5 of 63 below the benchmark, all submitted with train_pass; output tokens 0.81x the benchmark's on the
  same tasks.

## Caveats

- **The benchmark draws disagree by up to 3 pp** (E052 A 76.5, acc_manh3 77.1, run 7 74.0 on 114 tasks). Benchmark
  choice moves the gap by 1-4 pp.
- One replica draw per task: at n = 63-170 the CIs are ±6-9 pp.
- **Stack differences inside the benchmark.** acc_manh3 ran Manh's fork b9ad670a6 (E062's port). V12 runs Manh's
  current production server: wheel g8cf0c0664b, no DCP, a different offload connector, batch-sharded sampling. B32-M V3
  runs his Kaggle wheel g5e1bccd07c with the 2r patch.
- Pod clocks include the notebook install: about 5 min for V12 and about 15 min for B32-M V3, whose server was up at
  0.24 h.

## Re-run

```bash
S=experiments/E062_manh_vllm_fork/per_layer/scripts/replica_vs_bench.py
W=experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_idmap.json
E=experiments/E052_next_code_gate/runs/new_ar/out/dsv41_logs/attempts.jsonl
A=experiments/E062_manh_vllm_fork/acc48/runs/acc_manh3/out/dsv41_logs/attempts.jsonl
R7=evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all.jsonl
.venv/bin/python $S manh_v12=experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data \
  b32m_v3=experiments/E062_manh_vllm_fork/per_layer/pod3_runs/b32m_v3/data \
  candBf_x2t=experiments/E055_kaggle_replica/runs/candBf_x2t \
  manh_x2t_w80=experiments/E062_manh_vllm_fork/l4/runs/manh_x2t_w80 \
  b32r1_x2t=experiments/E062_manh_vllm_fork/l4pod2/runs/b32r1_x2t \
  --bench e052a=$E --bench acc_manh3=$A --idmap $W --prefix-m 99 [--at-n 63,170] [--json out.json]
# sensitivity: --bench run7=$R7 alone, or all three --bench; null: pass $A (or $E) as the "replica" with the others as --bench
```
