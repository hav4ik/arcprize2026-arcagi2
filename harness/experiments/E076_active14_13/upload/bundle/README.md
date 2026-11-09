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

**Private experimental preview, E076 `seed14`, 2026-10-31. Full ARC evaluation of
this schedule is pending.** This checkpoint keeps 128 stored routed experts in
each of 40 layers and activates 4 in 28 layers and 6 in 12 layers. The shared
expert remains active. Mean routed top-k is 4.6.

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

## Evidence so far

On 24 paired E065 training families, generated-token NLL increases by 1.057%
for images and 1.213% for text, averaging 1.135% relative to k6. Tool-block NLL
increases by 3.746% and 4.105%, respectively. The initial uniform-k4 (~13.17B)
candidate has a larger average generated-token NLL increase of 3.153%.

Short generated diagnostics on **12 Kaggle ARC training originals, pass@1,
one draw, 24k base output budget**: seed14 100% image / 100% text; k6 baseline
100% image / 83.3% text; uniform-k4 83.3% image / 91.7% text. These are small
training-only diagnostics, not evidence of improved or preserved evaluation
accuracy. The broader E076 architecture search and full 120-task validation are
still running. See `evaluation/seed14_preview.json` for the exact measurements.

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
