# E074: reduce active experts in the causal image E128 model

Started 2026-10-31 UTC (2026-10-30 Pacific). User requests active-parameter pruning,
initially approximately 16B → 14B / 12B, then confirms **remove one on average (mean top-k 5)**, allowing sensitive layers
to retain six. Target: at most 2 percentage points of task-weighted pass@1 loss.

## Baseline and prior evidence

- Model: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`
  at `209c810a747c043c2aca474843de8e357f920ef6`.
- E066's **fixed causal image** re-calibration, 128 stored routed experts in each
  of 40 layers, six active routed experts plus one shared expert per layer.
- Original 64 GiB compact Engram table. The earlier image-E128 HF repository now
  returns 404 even with the full-access credential. Use the same tool table from
  B32 at `9bce98c5ecda0a9226edc6eab72a689089db1e37`, in its three-file packaging;
  require identical Engram config and replace only table entries in the index.
- Image serving must use `is_mm_prefix_lm=false` and `--disable-chunked-mm-input`.
  E066's earlier bidirectional image results are not a valid baseline.
- E066's "dynamic count" candidate was static **stored** expert counts per layer
  (72–152, average 120); active top-k remained six. It improved dev NLL but lost
  ARC accuracy. Its sensitivity curve therefore cannot directly select a top-k
  schedule, and NLL alone is insufficient for accepting a candidate.
- E062 measured extra Humming repacking allocations for distinct stored expert
  counts. Keeping E128 everywhere avoids changing those weight tensor shapes;
  top-k-dependent execution workspaces and graph memory still need measurement.

## Parameter accounting

Each routed expert has `3 × 5120 × 2304 = 35,389,440` logical parameters.
One fewer active expert in all 40 layers removes exactly **1,415,577,600** routed
parameters per decoded token. The total below uses the *rounded advertised 16B*
baseline, not a newly measured exact parameter count. It excludes vision-encoder
work, and should not be equated to whole-system FLOPs, VRAM savings, or speedup.

| Uniform routed top-k | Active routed parameters removed | Approximate total decode active |
|---|---:|---:|
| 6 | 0 | 16B |
| 5 | 1.416B | 14.584B |
| 4 | 2.831B | 13.169B |
| 3 | 4.247B | 11.753B |

A mean-five schedule has 200 active routed slots across 40 layers, versus 240 at
baseline. Using only 4/5/6 gives one layer at four for every layer protected at six.
A mean-one schedule with some layers at six requires **zero** routed experts in
other layers; that is a different experiment involving routed-branch skipping.

## Implementation and tests

Use the E043 pinned `vision11` fork in a separate `/workspace/dsv4/venv` on Vast
58489667 (4× RTX PRO 6000 WS; $5.649705/hour). Existing Runpod workloads remain
owned by their experiments.

`scripts/apply_topk.py` changes two files in that isolated install:
1. allow E128 routing for k=1..6 using the existing generic Triton kernel;
2. read one top-k value per layer before constructing its MoE/router/workspaces.

`E074_TOPK_FILE` supplies the schedule. Restart the engine for each schedule so
CUDA graphs and workspace shapes agree. We preserve native sqrt-softplus scores,
selection-only text/image biases, normalization and the 1.5 routed scale.
No expert weights, shared experts, attention, vision or Engram are pruned here.

GPU check on 58489667: **36/36 pass** for k=1..6, batch sizes 1/17/129,
text/image positions and exact ties, compared with independent Torch selection.
The full quantized Humming path also passed at uniform k=6 and k=5, with CUDA
graphs, real text/image requests, exact token alignment and generation. A mixed schedule (layer 0 at k=4, all others at k=6) also passed both
modalities and generation; the sweep has advanced to layer 1.
No new CUDA/Triton kernel has been written.

Token-adaptive k would need genuine dispatch skipping, correct renormalization
and graph-safe ragged/padded execution; setting weights to zero is not proof of
less compute. Full-layer pruning additionally needs a CED/KV/index dependency
audit and correct handling of delayed mHC mixing and final collapse.

## Data and evaluation plan

`scripts/prepare_pilot.py` selects 32 paired E065 image/text rollouts from the
existing family-held-out development split, at most 30k tokens each (seed 74).
The parent IDs are asserted to be ARC training IDs and disjoint from all 120
Kaggle evaluation IDs. Text twin batch names are normalized only for pairing.
The generated manifest records the exact families. The short-rollout subset is
a pilot, not evidence of long-context or ARC equivalence.

Confirmed first batch (mean top-k five):
1. Baseline identity and full-model kernel smoke.
2. Uniform mean-budget baseline and small per-layer sensitivity screen on the
   *already quantized causal E128*, not the original full model.
3. Allocate the fixed total slot budget with the measured per-layer costs.
   `scripts/budgets.py` supplies a multiple-choice knapsack; its additive loss
   estimate must be checked by actually evaluating the combined schedule.
4. Check a fresh training-family subset and real generated tool trajectories.
5. Only after selection, matched baseline/candidate solving on the **120 Kaggle
   evaluation task versions**, task-weighted pass@1 (primary), pass@2 (secondary),
   task-cluster bootstrap, token cost, failures, and end-to-end speed. Keep the
   prompt, image handling, effort, budgets, sampling and cache policy fixed.
6. Report noninferiority only if the paired uncertainty supports the agreed
   margin; failure to detect a difference is not evidence of equivalence. A
   single 120-task comparison may not resolve a two-percentage-point margin.

All self-hosted tool runs retain prefix caching and log cache hits, KV occupancy
and preemptions. No full-layer removal or repair training is launched yet.
If top-k pruning needs repair, use training-only teacher distillation with the
causal E128 as teacher, verifying that routed-expert updates reach the export
(E069's exported routed weights were unchanged despite the SFT description).

## Budget and status

The user authorized continuing on the rented box and topping up if more hours
are needed. First batch is bounded to 14 box-hours (about $79.10), including
40 single-layer k=4 sensitivity screens, a uniform k=5 control, a mean-five
4/6 schedule, joint checks on 32 families, and two matched ARC arms if time permits.
The batch reports a stopped state if insufficient time remains for a complete arm.
The 4 screening families are a subset of the 32 verification families; 28 are fresh.
No new instance is rented; the rental is never stopped or destroyed automatically.

Status: the 40-layer sweep and all three 32-family verification arms are complete.
The profiled mean-k5 schedule was selected by training NLL. The short 12-task
Kaggle image/tool check scored pass@1 58.3% / 66.7% / 75.0% for k6 / uniform k5 /
profiled mean-k5 (one draw, 24k base budget; diagnostic only). The 120-task x2
image/tool baseline and candidate both completed. Full results are below;
the 2pp noninferiority target remains inconclusive.
The local `scripts/monitor.py` synchronizes results each minute, scores completed
ARC arms, and updates the registry; it has a 16-hour bound. Actual cgroup memory limit is 512.79 GiB
(the host-wide `free` display reports 755 GiB). Uniform k=5 Humming/model compatibility has passed. Process cleanup uses captured identities and polling because this
Vast host rejects psutil's pidfd-based wait with EINVAL.

Research sources and method comparison: [prior-work review](../../prior_works/moe_active_parameter_pruning.md).

### Harness and modality audit (2026-10-31)

`analysis/harness_audit.json` records the checked scope. All 41 frozen harness
files still match the current local `evals/` files; main harness SHA256
`914c849b20715fc57c2e10a9161685654ad2e350e2479efefdcb859c680addd1`.
Live ARC uses sub80s40 + minimal header, multi-test 0.5, train checks, fallback,
rescue and auto-repair, images, causal attention, context 163840 and local tools.
This is the E043 vision11 serving fork and chat image path, **not** Manh's newer
V12/PR3 token-exact image path. Current local harness and latest serving stack
are separate claims.

Pruning calibration/NLL replays use **both** image+tools and text+tools E065
traces (32 paired families; all 64 traces contain tool responses). They were
collected with E046-best/E058 settings on 2026-10-28 and reused, not freshly
generated with the latest self-hosted harness changes. Both modalities contribute
equally to schedule selection. Fresh generated-answer checks in E074 so far,
including the full 120-task comparison, are **image+tools only**. Text-only
task-accuracy retention has not yet been measured for the active-k candidate.

Prior E128 ARC runs are retained as historical references. The new k6 control
holds checkpoint, server fork, hardware, harness, modality and Kaggle task
versions fixed with the active-k candidate; historical cross-setup differences
are too large to substitute directly for this experiment's 2pp comparison.

## Initial smoke results (4 training families, paired image/text)

| Schedule | Text generated-token NLL | Image generated-token NLL |
|---|---:|---:|
| k=6 | 0.820117 | 0.780277 |
| k=5 | 0.824182 | 0.786556 |

Relative NLL change: +0.50% text, +0.80% image. This small short-rollout screen
is encouraging but does not measure ARC pass@1 or establish the 2pp target.
Startup was 335s cold / 95s warm; these are not a model speed comparison.

Operation: `supervisorctl status e074` on Vast 58489667. The service has no
autorestart. Main log `/workspace/dsv4/logs/batch.log`; remote artifacts
`/workspace/dsv4/repo/experiments/E074_active_params/results/`; matching local
`results/` are synchronized by the monitor. `results/batch_status.json` records
the current phase or failure. `scripts/score_arc.py` writes the final
`results/arc_report.json` locally using the Kaggle answer key.

## Layer statistics

The reproducible collector is `scripts/layer_stats.py`. Run with system `python3`
(NumPy and Matplotlib installed) and `MPLCONFIGDIR=/tmp/e074-matplotlib`; the
project venv supports `--no-plot`. `--watch-hours 16` refreshes local artifacts
when the existing result monitor downloads another completed arm. It neither
changes nor sends requests to the inference server.

- [Chart](analysis/layer_stats/layer_stats.png) and [PDF](analysis/layer_stats/layer_stats.pdf):
  all 40 historical layer distributions plus the current active-k loss curve.
- [Layer CSV](analysis/layer_stats/layer_stats.csv): historical distribution metrics,
  old stored-expert sensitivity, and current k6→4 sensitivity in separate columns.
- [Per-class CSV](analysis/layer_stats/routing_by_class.csv): separate image/text
  traffic and prompt/reasoning/tool-block/image positions, plus the E066 weighting.
- [Paired-family CSV](analysis/layer_stats/paired_family_deltas.csv): image/text
  generated and tool-block NLL changes for each training family.
- [JSON with provenance and limitations](analysis/layer_stats/layer_stats.json).

Historical input is E066's **causal full-384 model** census: 24,043,161 scored
positions, before E128 selection and GPTQ. For the pruning recipe, class weights
are `[1,1,10,1]` in layers 0–19 and `[0,1,10,1]` in layers 20–39. Expert IDs refer
to that original model. Do not present these distributions as current E128 routing.

The census confirms substantial layer heterogeneity. For example, the effective
expert count (exp entropy) under amplitude weighting is 164.7 in layer 1 and 43.9
in layer 25. Layers 36–39 need special care: their largest-amplitude experts hold
57–80% of the sum of weighted expert-output norms, but are selected at only
0.56–1.87% of weighted positions. These rare massive activations distort norm
concentration; the frequency-based effective counts there remain 141–184. Neither
aggregate count tells us how many of the six experts matter on an individual
token, and individual output norms do not account for vector cancellation.

The **current** measurement instead tests each layer at k=4 while keeping all
other layers at k=6 in quantized causal E128. Its plotted intervals resample paired
image/text training families together, preserving the image/text-average,
token-weighted NLL statistic. Four families provide an initial ranking only:
intervals exclude neither numerical variation between server runs nor unknown
task families, and are not corrected for testing 40 layers. Tiny negative changes
are not grounds to claim an improvement. The combined schedule gets the planned
32-family check and matched Kaggle task-weighted pass@1 comparison.

## Short generated-answer checks

User requested short evals during the experiment. Each of the three 32-family
verification arms now reuses its loaded engine for the **same 12 Kaggle evaluation
tasks, one attempt per task**, before the full 120-task runs. This adds no engine
restarts. `scripts/short_eval.py prepare` fixes the task IDs by a seeded hash,
stratified by maximum grid area and single/multiple test inputs, using challenges
only. The exact subset is in `data/short_eval_manifest.json`; no answer key is sent
to the GPU instance.

The short diagnostic uses the image/tool harness with a **24k base output budget**,
12k soft budget, 20 turns, multi-budget 0.5, context 65,536, 12 workers and one
attempt. Baseline k6, uniform k5 and the profiled mean-k5 schedule get identical
settings and task IDs, separate cache salts, and prefix/cache metrics. These are
short-budget pass@1 results, not estimates at the full 80k budget or evidence for
a 2pp noninferiority margin. Candidate selection remains training-NLL-based.
The existing 35-minute verification-arm timeout and 14-hour batch bound still
apply; worst-case requested output is recorded by the task manifest and budgets.

Preflight passed on Vast: exact frozen-harness CLI dry run and an actual
LocalExecutor session importing NumPy/SciPy/SymPy/scikit-image and transforming a
test grid. The frozen prompt asset and missing client/tool dependencies were added
before ARC execution. Exact versions are saved in `results/tool_environment.freeze`,
and the receipt is `results/short_preflight.json`. The selected subset's nominal
output budgets sum to 360k tokens per arm including multi-test scaling.

The local `scripts/short_eval.py watch --hours 16` scores each arm as the result
monitor downloads it. `results/short_arc_report.json` records task-weighted
pass@1, paired task uncertainty, better/worse tasks, missing predictions, errors,
stop reasons, output tokens and elapsed time. Full results also report the 108
tasks outside this early diagnostic separately. All current `layer*_k4` screens
finish as planned before these generated-answer checks run.

## First full ARC result (2026-10-31)

Both arms finished on the **120 Kaggle-version evaluation tasks, two attempts
per task**, images + tools, full 80k base output budget. Each file contains 344
test-input records (172 pairs x2), corresponding to all 240 task attempts, with
zero evaluation error records. Missing answers count as incorrect. All tasks
have equal weight; pass@1 averages the two draws and each task's test inputs.

| Metric | E128 k6 baseline | Profiled mean-k5 |
|---|---:|---:|
| Task-weighted pass@1 | 72.43% | 70.83% |
| Task-weighted pass@2 | 83.47% | 81.81% |
| Output tokens / task attempt | 38,905 | 39,483 |
| Evaluation wall time | 2h 47m 35s | 2h 39m 58s |
| End-to-end output tokens / second | 928.6 | 987.3 |

Pass@1 difference is **−1.60 percentage points**, with paired task-bootstrap 95%
interval **[−7.01, +3.54]** (5,000 draws). The point estimate is inside the agreed
2pp margin, but the interval does not establish noninferiority. The 108 tasks
outside the short diagnostic give −2.47pp [−8.10, +3.47]. Candidate selection was
based on training NLL, not the short ARC scores.

Observed token throughput rose **6.3%**, total wall time fell **4.5%**, and output
tokens increased **1.5%**. This is one sequential same-box pair of real tool runs;
token counts, sequence lengths and tool trajectories differ, so it is not an
isolated kernel speed measurement. Active parameters fell by approximately 8.8%
using the rounded 16B anchor. The gain is modest, and text-only ARC retention
remains unmeasured.

The full remote batch finished at about 12:37 UTC after 7.65 box-hours (~$43.22
at the quoted rate), with the rental left running. Local sync had stopped
updating at 07:09 UTC; both completed remote result sets were retrieved and scored
locally at 13:38–13:42 UTC. A Luna monitoring sub-agent independently checked
completeness and score arithmetic. No rental was stopped or deleted.

Machine-readable results: [full report](results/arc_report.json),
[baseline](results/baseline_arc/arc.jsonl), [candidate](results/candidate_arc/arc.jsonl).
