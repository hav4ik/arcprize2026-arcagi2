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

Status: baseline and uniform k=5 full-model screens passed; the 40-layer k=4
sensitivity sweep is running under supervisor service `e074`. No ARC result yet.
The local `scripts/monitor.py` synchronizes results each minute, scores completed
ARC arms, and updates the registry; it has a 16-hour bound. Actual cgroup memory limit is 512.79 GiB
(the host-wide `free` display reports 755 GiB). Uniform k=5 Humming/model compatibility has passed. Process cleanup uses captured identities and polling because this
Vast host rejects psutil's pidfd-based wait with EINVAL.

Research sources and method comparison: [prior-work review](../../prior_works/moe_active_parameter_pruning.md).

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
