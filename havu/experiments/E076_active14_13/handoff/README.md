# Active expert pruning results and SFT handoff

This handoff records the completed E076 image+tools evaluation and the information
needed to apply or recalibrate its active-expert schedule on a compatible SFT
checkpoint. The ~14.02B schedule keeps every E128 weight and changes how many
routed experts each language-model layer executes. Its image+tools pass@1 is
0.76 percentage points below E128 and 0.83 points above the earlier ~14.58B
schedule. It runs 4.39% faster than E128 end to end, and takes essentially the
same time as ~14.58B because it generates more tokens. The text+tools comparisons
are still running at this snapshot; their results must be added separately.

The schedule is transferable as a configuration when architecture and serving
support match. Accuracy transfer to the teammate's SFT weights has not been
measured. Revalidate against the same SFT checkpoint with the original routing.

## Completed image and tools results

Kaggle ARC-AGI-2 evaluation version, **120 tasks, 172 test inputs, two draws per
task**. This is 240 logical attempts, stored as 344 per-test-input records.
Pass@1 below averages both draws and gives each task equal weight; multi-input
tasks receive fractional credit across their test inputs.

| Model and routing | Mean routed k | Task pass@1 | Task pass@2 | Output tokens | Output tok/s | Eval seconds |
|---|---:|---:|---:|---:|---:|---:|
| E128 original routing | 6.0 | 72.4306% | 83.4722% | 9,337,084 | 928.6467 | 10,054.5066 |
| E074 profiled ~14.58B | 5.0 | 70.8333% | 81.8056% | 9,475,978 | 987.3087 | 9,597.7868 |
| E076 seed14 ~14.02B | 4.6 | 71.6667% | 83.8889% | 9,771,962 | 1,016.4742 | 9,613.5860 |

For ~14.02B versus E128: pass@1 **-0.7639 pp**, paired task-bootstrap 95% interval
**[-6.0417, +4.3750] pp**; output throughput **+9.4576%** and eval time **-4.3853%**.
Versus ~14.58B: pass@1 **+0.8333 pp**, interval **[-3.7500, +5.4167] pp**;
throughput **+2.9540%** and eval time **+0.1646%**. The point estimate satisfies
the requested <=2 pp loss relative to E128; the interval does not establish that
margin. These are reused, matching E074 controls, not fresh simultaneous
replicates. No new schedule was selected using these ARC scores.

Times exclude engine startup and include the real tool harness through its final
attempt. They are measurements on four RTX PRO 6000 GPUs, not L4/Kaggle speed
claims. Active parameters, output tok/s, and total task runtime are distinct.

### Tokens tools and failures on the complete set

These counts deduplicate multi-test records by `(task_id, attempt)`, retaining
the pair-zero counters. Raw CLI summary totals can double-count tools because
each test-input record repeats the attempt's tool counters.

| Metric | E128 | ~14.58B | ~14.02B |
|---|---:|---:|---:|
| Mean output tokens per attempt | 38,904.52 | 39,483.24 | 40,716.51 |
| Mean tool calls per attempt | 17.8500 | 17.7625 | 18.0583 |
| Tool calls total | 4,284 | 4,263 | 4,334 |
| Tool errors or rejections after repair | 545 (12.72%) | 555 (13.02%) | 587 (13.54%) |
| Attempts receiving automatic repair | 91 | 109 | 105 |
| Format rescues | 2 | 2 | 1 |
| Token, turn, or context cap stops | 12 | 6 | 9 |
| Attempts missing the pair-zero answer | 9 | 4 | 6 |
| Server or harness errors | 0 | 0 | 0 |

Tool errors include Python execution failures, rejected/malformed tool calls,
and skipped truncated calls, counted after automatic repair. They are different
from server/harness failures. A cap stop can still yield a fallback answer.
Training-check success is also different from evaluation correctness.
Full counter definitions and totals are in [full_usage.json](evaluation/full_usage.json).

## Exact schedule and parameter accounting

The language transformer uses 40 layers, indexed **0 through 39**, with hidden
width 5120 and expert intermediate width 2304. Each layer retains 128 stored
routed experts and one shared expert. Native routing chooses expert identities
per token; the **count** is static per layer. Attention, vision, Engram, shared
experts, and expert weights are unchanged by this pruning operation.

The default [active_experts.json](schedules/active_experts.json) is:

```json
[6,4,4,4,4,4,4,4,4,6,6,6,6,4,6,4,6,6,6,4,4,4,4,4,4,6,4,4,4,4,6,4,4,4,4,4,4,6,4,4]
```

Six-expert layers: **0, 9, 10, 11, 12, 14, 16, 17, 18, 25, 30, 37**.
The remaining 28 layers use four. Total routed slots across layers: 184; mean
top-k: 4.6. This retained set is a conservative starting point for transfer, not
proof that every member must use six after a different SFT run.

Each routed expert contains `3 * 5120 * 2304 = 35,389,440` logical parameters.
Relative to k6 in all 40 layers, seed14 removes 56 routed slots, or exactly
**1,981,808,640 active routed parameters per decoded token**. With the rounded
advertised 16B baseline as anchor, the approximate decode-active count is
**14.01819136B**. The shared expert is not part of the routed top-k count.
The estimate excludes vision-encoder work and does not imply stored-weight or
VRAM savings. All 128 experts per layer remain on disk.

| Schedule file | Slots | Mean routed k | Approximate active parameters |
|---|---:|---:|---:|
| [baseline_k6.json](schedules/baseline_k6.json) | 240 | 6.0 | 16B |
| [e074_mean5_14p58B.json](schedules/e074_mean5_14p58B.json) | 200 | 5.0 | 14.5844224B |
| [active_experts.json](schedules/active_experts.json) | 184 | 4.6 | 14.01819136B |

The ~14.58B alternative is a profiled 4/6 schedule, not uniform k5. Use its exact
JSON rather than replacing every layer with five.

## Checkpoint identity and unchanged weights

The private published model is
[chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview](https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview).
The original verified upload is revision
`87059808afa6cf3f3fe928ad80749de6f244f5bd`. The subsequent schedule-only copy commit
is recorded in [hf_schedule_copy_receipt.json](provenance/hf_schedule_copy_receipt.json).
Root `active_experts.json` and `schedules/active_experts.json` are byte-identical.

Model weights came from:

- Causal image E128:
  `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`
  at `209c810a747c043c2aca474843de8e357f920ef6`.
- Compatible compact Engram table:
  `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`
  at `9bce98c5ecda0a9226edc6eab72a689089db1e37`.

The upload contains 46 safetensors files totaling 132,053,122,744 bytes, including
Engram. Every copied weight file was checked against its source LFS SHA256 or Git
blob hash, and metadata was downloaded and checked by SHA256. The full per-file
map is [candidate_manifest.json](provenance/candidate_manifest.json); the combined
[weight index](provenance/model.safetensors.index.json) maps tensors to those files.
The weights themselves are not duplicated in this handoff archive.

## Method actually used

1. **E074 ranking:** on the already quantized causal E128, reduce one language
   layer at a time from six to four active experts, leaving the other 39 at six.
   Measure teacher-forced generated-token NLL on **four paired E065 training
   traces per modality**. Average token-weighted image and text NLL, subtract the
   k6 baseline, then sort by `(delta, layer_index)`. This small ranking screen is
   an acknowledged source of noise. Exact costs and trace identities are bundled.
2. **E074 ~14.58B:** reduce the lowest-cost 20 layers to four, keep 20 at six.
   Compare this combined schedule and uniform k5 on 32 paired E065 training
   families; choose the profiled schedule by average generated NLL. Four of the
   32 families informed the screen, so 28 are fresh for that joint check.
3. **E076 seed14:** use the same ranking and reduce its lowest-cost **28** layers
   to four, keeping 12 at six. This exactly reconstructs the uploaded 184-slot
   schedule. Check the combined schedule on E076's **select24** pool in both
   modalities, plus short generated tool diagnostics on 12 training originals.
4. Freeze the uploaded seed14 schedule before its 120-task evaluation. The user
   prioritized full tool evaluation; further E076 architecture search was
   deferred. Completed new profiles are all 40 single-layer k4 screens and the
   first six k5 screens. **The larger k3/k4/k5 joint search was not completed,
   and seed14 did not receive the independent audit32 run.** The baseline audit
   exists; it must not be presented as a candidate audit.

On select24, seed14 relative generated NLL increases were **+1.0569% image** and
**+1.2125% text**; tool-block NLL increases were **+3.7464%** and **+4.1050%**.
These are teacher-forced training-domain loss changes, not task accuracy losses.
The 12-task generated training diagnostic was 100% image/text for seed14 and
100%/83.33% for k6; it is not validation-set evidence.

The deferred optimizer in `scripts/e076/search.py` supports measured k3/k4/k5/k6
curves, exact-slot dynamic programming, and joint slot-swap checks. Its proposed
selection objective is mean relative generated NLL + 0.25 * mean relative
tool-block NLL + 0.25 * worst-modality generated NLL. That optimizer did **not**
select the evaluated seed14. Single-layer costs are a proxy; always measure a
combined schedule rather than assuming effects add.

## Calibration data and contamination boundaries

Source family: E065 paired image+tools and text+tools rollouts, available in the
private dataset `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e065`.
Exact local parquet bytes are identified by source SHA256 in
[source_provenance.json](calibration/source_provenance.json); a pinned HF dataset
revision was not recorded, so do not substitute today's dataset contents without
checking identity. Text batch prefixes `e065t_` normalize to `e065_` only for
pairing `(batch, task_id, pair, attempt)`.

E076 prepared 72 disjoint training-parent families with random seed 76:

| Pool | Paired traces per modality | Image tokens | Text tokens | Selection rule |
|---|---:|---:|---:|---|
| screen | 16 | 251,101 | 220,843 | Both modalities <=30k tokens |
| select | 24 | 645,939 | 583,358 | <=60k; alternate random and longer traces |
| audit | 32 | 898,082 | 800,470 | <=60k; excludes all E074 families |

All parent IDs belong to ARC training; overlap with Kaggle evaluation parents is
zero. `nvarc_full` and `arc2_evaluation6` must remain excluded. Exact families,
rollout keys, prepared token/role arrays, and the corresponding message/tool
parquets are in [calibration/prepared](calibration/prepared/). Every prepared
file has a recorded hash. `roles == 1` identifies generated targets; tool-block
loss covers generated tokens from `</think>` through EOS. The NLL runner checks
every offline token against the server's returned prompt-logprob stream.

For the new SFT checkpoint, check overlap against **its own training manifest**.
The current calibration/audit split is held out from this pruning procedure; it
is not automatically held out from the teammate's SFT. Rebuild audit families if
she trained on them. Retokenize if tokenizer, template, image processing, or
tool serialization changes; never reuse mismatched token IDs silently.

## Frozen tool evaluation and scoring

The 44 bundled frozen harness files match E074 byte for byte. Their hashes are
in [frozen_harness_manifest.json](provenance/frozen_harness_manifest.json).
This is the E043 vision11 / frozen E074 tools stack, **not Manh's newer V12
token-exact-image stack**. Auto-repair remains on in every compared arm.

Key settings are `lean2sub`, grid answers, `submit_answer`, session fallback,
training checks, code rescue, multi-test mode, 64 workers, two attempts,
temperature 1, top-p 1, thinking enabled with reasoning effort 50 and the minimal
effort header. The per-turn output cap is 64k, base total output 80k, soft budget
40k, and base turn cap 40. Multi-input budgets scale by `1 + 0.5*(n_test-1)`.
Context limit is 163,840. Images use `text+image` and image-cell 0; text uses JSON
grids and `--token-exact`. **Never use that text-only token-exact path for images.**

The wrapper `scripts/e066/b32_local_eval.py` loads the harness from its working
directory. It uses the local IPython executor, 4096 MiB per session and 7200 CPU
seconds. Run it from `frozen/evals`; it is not the HTTP sandbox entry point.
Exact serving and eval argv arrays are in each `evaluation/<model>/` directory.
The image candidate required command equality with E074, allowing only port,
base URL, output path, and cache salt to differ. Use a unique cache salt for each
run. The primary image checkpoint weights, prompt, budgets, sampling, causal
image attention, and cache policy were held constant.

For task t with n test inputs and draws a=0,1, pass@1 is the mean correctness
over its `2*n` answers, then averaged equally across 120 tasks. Pass@2 uses the
OR across draws separately for each test input, averages those n inputs, then
averages tasks. The paired bootstrap resamples **tasks**, 5000 replicates with
Python Random seed 0, taking sorted entries 125 and 4875. This is not selected
first-answer accuracy, majority vote, or requiring every test input to be right.
The scorer is `scripts/e066/score_b32_engram.py`; its original workspace paths
must be adapted when used outside this repository.

Kaggle challenges are included; evaluation solutions are **not**. Their SHA256
and the challenge hash are in [metadata.json](metadata.json). Keep solutions
local to scoring and out of calibration, SFT, and the inference host. The GitHub
ARC evaluation copy differs on six tasks and is not interchangeable.

## Serving the three schedules

The tested server is vLLM
`0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`, Torch `2.13.0+cu130`,
Triton `3.7.1`, Humming kernels `0.1.12`, on four RTX PRO 6000 Blackwell
Workstation GPUs with driver `590.48.01`, 97,887 MiB reported memory per GPU and
450 W power limits. Full Python/package versions and patched source hashes are
in [environment.json](serving/environment.json).

The included patch extends the existing E128 router support to k1 through k6
and chooses each layer's count **before** constructing MoE workspaces and CUDA
graphs. It preserves native sqrt-softplus routing scores, selection-only
text/image biases, selected-score normalization, and routed scale 1.5. The 36
GPU router reference cases passed, followed by mixed-k real-model serving checks.
No new MoE GEMM kernel was written for this tested stack.

After installing the matching E043 environment and existing compact Engram/
Humming support, choose exactly one schedule before starting the server:

```bash
export MODEL_DIR=/absolute/path/to/downloaded/model
export VLLM_PACKAGE_DIR=/absolute/path/to/site-packages/vllm
python "$MODEL_DIR/serving/apply_active_topk_e043.py" "$VLLM_PACKAGE_DIR"

# Choose ONE of these assignments before starting a fresh engine.
export E074_TOPK_FILE="$MODEL_DIR/schedules/active_experts.json"          # ~14.02B
# export E074_TOPK_FILE="$MODEL_DIR/schedules/e074_mean5_14p58B.json"    # ~14.58B
# export E074_TOPK_FILE="$MODEL_DIR/schedules/baseline_k6.json"         # ~16B

export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export DSV41_EXPERT_HUMMING='{"quant_method":"humming","weight_dtype":"uint2","weight_scale_dtype":"bfloat16","group_size":128,"has_zero_point":true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size":64}'
```

Use the captured serving argv, adapting only model location and per-run addresses.
It uses TP4, FP8 KV, block size 128, full CUDA graphs, max sequences 64, batched
tokens 4096, utilization 0.80, compact Engram CPU offload, prefix caching and the
DeepSeek tokenizer/reasoning/tool parsers. Image runs require
`--hf-overrides '{"is_mm_prefix_lm":false}' --disable-chunked-mm-input`.
Log prefix hits, KV usage and preemptions during real multi-turn traffic.

Changing the environment variable on a running process does not switch its
schedule. Restart the engine and verify startup logs for every layer against
the chosen JSON. The HF config's `num_experts_per_tok_per_layer` is metadata for
compatible loaders; **the tested E043 patch reads E074_TOPK_FILE**. The scalar
top-k stays at six for compatibility. A server that ignores the list/file can
silently execute the original ~16B routing.

For Manh's `g22df163d2d` fork, the port is **not yet GPU validated**. Its model
initialization can pass per-layer active counts, but `dsv4_topk.py` has fixed-six
guards, loops, strides, allocations and warmup shapes, and its caller must pass
top-k. See [Manh audit](serving/manh_audit.json) and
[serving notes](serving/ACTIVE_EXPERTS.md). Validate k6 identity, text/image bias,
ties, padding, mixed-k Humming with CUDA graphs, and realistic prefix-cached tool
traffic on the target GPU before claiming support. The stored-expert-count patch
alone does not provide active-top-k support.

## Applying this to the teammate SFT checkpoint

1. Record the SFT model ID, immutable revision or file-manifest hash, parent
   checkpoint, training-data manifest, tokenizer/template revision, merge state,
   quantization settings, and expert/attention/router/vision/Engram components
   actually updated. Verify that the exported tensors contain the intended SFT
   changes; an earlier project SFT exported unchanged routed experts.
2. Confirm 40 language layers, hidden width 5120, expert width 2304, E128 stored
   routing, one shared expert, and compatible layer mapping. An E256 checkpoint
   or different expert width requires a separate adaptation; do not call it this
   E128 transfer. Preserve the SFT checkpoint's own weights, tokenizer, config,
   vision and Engram. **Do not replace its tensors or config wholesale with the
   original preview's files.**
3. Create a routing overlay from the SFT config and this exact schedule. The
   helper validates dimensions, leaves the source untouched and refuses an
   existing destination. It writes no weights and makes no accuracy claim:

```bash
python scripts/e076/sft_schedule_overlay.py \
  --source-config /path/to/SFT/config.json \
  --source-id teammate/SFT-checkpoint \
  --source-revision IMMUTABLE_REVISION_OR_MANIFEST_HASH \
  --schedule schedules/active_experts.json \
  --output-dir /path/to/new-SFT-routing-overlay
```

4. On the tested E043 stack, serve the **SFT model directory** and point
   `E074_TOPK_FILE` at the overlay schedule. Point `E043_GPTQ_DIR` at the SFT's
   quantized experts if it uses the same GPTQ layout; BF16/other quantization
   requires its own supported loader configuration. The overlay directory alone
   is not a loadable model. Do not use `batch.py` unchanged: it hardcodes the
   original `/workspace/dsv4/models/causal-e128` path. Create an isolated SFT
   experiment and replace all model/quantization/result locations explicitly.
5. Measure a fresh k6 baseline on the SFT checkpoint. Test this seed14 schedule
   and the previous ~14.58B schedule on the **same SFT weights**, same E065
   image/text traces and same serving stack. Inspect generated/tool NLL, tool
   formatting, repairs and a short training-only generation diagnostic.
6. If transfer is weak, recollect per-layer costs on the SFT checkpoint. Use
   separate screening, joint-selection and independent-audit families. Keep
   layers requiring six fixed at six; shrink only layers supported by evidence.
   Evaluate joint schedules because marginal effects interact. Keep audit and
   ARC outcomes out of architecture selection. The parameter count may be above
   the approximate target when protection requires it.
7. Freeze the SFT schedule before full evaluation. Prioritize **text+tools**, then
   images+tools, all 120 Kaggle tasks with two draws and matching settings. Report
   both within-SFT references (k6 and ~14.58B), plus clearly labeled historical
   base-model references. This separates SFT gains/losses from pruning effects.
   Report task-weighted pass@1/pass@2 and paired intervals, output tokens, tools,
   errors, no-answer/cap rates, prefix health, tok/s and whole-run time.

Fill [SFT_RUN_TEMPLATE.json](SFT_RUN_TEMPLATE.json) for the new checkpoint and
every compared routing arm. Null fields are intentionally unknown until the SFT
export is available; they are not defaults to reuse from this base-model run.

If recovery training is needed, keep its data and checkpoint revisions separate
and repeat the export checks. The current active-pruning result involved no
retraining. The future ~12B target is approximate; any layer that needs six stays
at six, without a parameter-budget override, and its evals are text+tools first.

## Artifact map and integrity

- [metadata.json](metadata.json): architecture, schedule, selection provenance,
  data hashes, scoring definition, reference identities and SFT conditions.
- [evaluation](evaluation/): full predictions, exact argv, timers, schedules,
  cache counters, final paired report and complete token/tool totals.
- [calibration](calibration/): exact paired family lists, training-only prepared
  data, source hashes, measured losses and partial sensitivity profiles.
- [frozen](frozen/): the exact 44 harness assets used for these comparisons.
- [serving](serving/): tested patch and router test, runtime versions/hashes and
  the separate Manh audit. GPU validation status is explicit.
- [scripts](scripts/): source algorithms and overlay helper. Original experiment
  runners retain their original paths for provenance; adapt them in a new
  experiment before executing them against SFT weights.
- [FILE_INDEX.json](FILE_INDEX.json): byte size, SHA256 and original workspace
  path for every copied/generated artifact.

From this directory, run `sha256sum -c FILES.sha256` to verify the bundle.
The calibration data and model provenance are private project materials; the
archive contains no credentials and no evaluation solution file. It is a frozen
image-result snapshot, not an automatic feed of the unfinished text evaluations.
The live source report remains `experiments/E076_active14_13/results/report.json`.
