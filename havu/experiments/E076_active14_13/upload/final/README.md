---
license: mit
library_name: vllm
tags:
- deepseek
- mixture-of-experts
- pruning
- gptq
- experimental
- arc-agi
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
---

# DeepSeek V4.1 Flash 1017 — causal image E128, ~14.02B active preview

**Private experimental checkpoint, E076 `seed14`; final tools evaluations completed
2026-10-31.** On all 120 Kaggle ARC-AGI-2 tasks x two draws, the default schedule
loses **3.06 percentage points of text+tools pass@1** versus E128 k6 and **0.76
points on image+tools**. Text is the primary acceptance modality; this run misses
the requested <=2pp loss target on its point estimate. Noninferiority is not
established. The checkpoint keeps 128 stored routed experts in each of 40 layers
and activates four in 28 layers and six in 12 layers. The shared expert remains
active. Mean routed top-k is 4.6; weights are unchanged.

The exact routed-parameter reduction is 1,981,808,640 per decoded token. Relative
to the rounded advertised 16B baseline, the estimated decode-active size is
14.018B. This excludes vision-encoder work and is not a total stored-parameter,
download-size, VRAM or proportional-speedup claim.

## Contents and provenance

The quantized weights are unchanged from the causal image E128 checkpoint at
`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`, revision
`209c810a747c043c2aca474843de8e357f920ef6`. This repository includes the compatible
three-file compact Engram table from the tool E128 checkpoint at revision
`9bce98c5ecda0a9226edc6eab72a689089db1e37`, exactly as used in E074/E076. The
combined weight index is included. File origins and hashes are recorded in
`candidate_manifest.json`.

This is a routing change, with no additional weight training or requantization.
The default schedule is `active_experts.json`. Calibration uses E065 image/tool
traces and their paired E065 text/tool traces, with ARC training parents only.
No evaluation task labels are used to select this schedule.

## Final evaluation

All rows use **120 Kaggle ARC-AGI-2 evaluation tasks, two attempts per task**, the
same frozen tools harness with auto-repair on, and task-weighted mean-draw
pass@1 / pass@2. Multi-test usage is deduplicated by task/attempt.

| Modality | E128 k6 ~16B | E074 ~14.58B | Default E076 ~14.02B |
|---|---:|---:|---:|
| Text + tools (primary) | 72.29% / 81.81% | Skipped by user | 69.24% / 79.44% |
| Image + tools | 72.43% / 83.47% | 70.83% / 81.81% | 71.67% / 83.89% |

Text pass@1 difference versus k6: **-3.06pp**, paired task-bootstrap 95% interval
**[-8.68,+2.78]pp**. Image difference: **-0.76pp**, interval **[-6.04,+4.38]pp**.
These intervals do not establish the <=2pp loss margin. The 14.58B text arm was
explicitly skipped; no text accuracy claim is made for that alternative.

On four RTX PRO 6000 GPUs, matching elapsed windows show generation throughput
**+2.61% for text** (minutes5-155) and **+7.97% for images** (minutes5-150) versus
k6. Full text evaluation takes **1.13% longer**, with **2.74% more output tokens**;
full image evaluation takes **4.39% less time**. Image generation throughput is
+2.89% versus14.58B in the matching window. These are real-harness observations,
not proportional active-parameter speedups or L4/Kaggle hardware measurements.
Text post-repair tool error/rejection rates are12.34% versus10.17% for k6;
missing-answer attempts11 versus8. All five evaluated arms have zero failed
server/harness requests and zero preemptions. The current default is a poor
tradeoff for this text harness; image retention does not establish text retention.

[Full token/tool/accuracy tables](evaluation/e076_final_tools.md),
[exact results and provenance](evaluation/e076_final_tools.json).

## Calibration evidence and methodology

Calibration used a mixture of original ARC1000 training tasks and NVARC
training-parent E065 image/text rollouts, with family-disjoint screen/select/audit
pools. Kaggle120 evaluation tasks and all derivatives are excluded from fitting
or schedule selection. Sensitive layers remain at six. The evaluated seed14
schedule was frozen before the full evaluation, using E074 layer ranking;
the broader joint search and independent candidate audit32 were not completed.

On24 paired E065 training families, generated-token NLL increased1.057% for
images and1.213% for text; tool-block NLL increased3.746% and4.105%. Short
12-training-task diagnostics were too limited to establish retention. These
are training-domain loss measurements, not accuracy-loss percentages. No
recovery training or SFT transfer experiment was performed.

## Code and reproduction metadata

The [final reproduction bundle](reproducibility/E076_final_2026-10-31/README.md)
contains the exact schedules, frozen harness, E074/E076 algorithms, tested
serving patch, all five final prediction files, cache/throughput logs,
calibration provenance and prepared inputs, literature review, and SFT transfer
instructions. It includes integrity verification and optional rescoring against
the owner's exact Kaggle solution file; solutions and credentials are excluded.

[Download code and metadata archive](reproducibility/E076_active14_final_code_metadata_2026-10-31.tar.gz)
([SHA256](reproducibility/E076_active14_final_code_metadata_2026-10-31.tar.gz.sha256)).
The earlier image-only handoff remains unchanged in the private
[GitHub experiment](https://github.com/hav4ik/arcprize2026-arcagi2/tree/main/havu/experiments/E076_active14_13).

## Alternative with completed evaluation

`schedules/e074_mean5_14p58B.json` activates an average of five routed experts,
approximately 14.584B active parameters. It uses these same stored weights.
On **120 Kaggle ARC-AGI-2 evaluation tasks, two draws per task, images + tools**,
task-weighted pass@1/pass@2 were 70.83%/81.81%, versus 72.43%/83.47% for k6.
The pass@1 difference is -1.60 percentage points, paired-task 95% interval
[-7.01,+3.54]. Noninferiority at a two-point margin is inconclusive.
These E074 results belong to the alternative schedule, **not the default seed14**.

## Serving requirement

**A per-layer active-top-k patch is required.** An unpatched server can ignore
`text_config.num_experts_per_tok_per_layer` and run the six-expert baseline.
The custom metadata alone does not activate this architecture.

`serving/apply_active_topk_e043.py` is the patch tested on our E043 vision11
experimental fork. Set `E074_TOPK_FILE` to this model's `active_experts.json`
before starting that server. Full details are in `serving/ACTIVE_EXPERTS.md`.
The retained `inference/` reference implementation has not been adapted to this
schedule; do not treat its default generation script as an active-14B runner.

**Manh's fork `g22df163d2d` still needs the small router/model port described in
the serving notes. This upload does not claim drop-in compatibility with it.**

The checkpoint retains the causal image path, native routing scores and biases,
shared experts, vision tower, attention and Engram. Use prefix caching for tool
traffic and a context limit matching the server. Restart the engine when changing
schedules so workspace shapes, kernel specializations and CUDA graphs agree.
