---
license: mit
---
# DeepSeek-V4.1-Flash-1017 · ARC tool use with images · E128 2-bit, causal image calibration (E066 re-prune)

DeepSeek-V4.1-Flash-1017 pruned to 128 of 384 routed experts per layer and quantized to 2 bits, for ARC-AGI-2 tool
use on Kaggle 4xL4 with the image harness (E058: the E046 tool harness plus one image per grid) and its text twin.
The vision tower is kept unchanged.

This replaces the Kaggle Model `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit` (E066 candidate 1). That model's
calibration ran on a vLLM fork that attends **bidirectionally** over image spans (`is_mm_prefix_lm = vision_n_layers
> 0`); DeepSeek's reference is causal (E068). Here every image-harness replay ran with the fix
(`--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input`): the expert statistics and the GPTQ
activations. The pipeline is otherwise candidate 1's: same rollouts, seeds, sizes and class weights. HF copy without the Engram
table: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`.

**Engram included:** the 64 GiB tool table (top-frequency rows, 35% / 32% of rows on layers 1 / 14) as the 16 row-chunk files `model-engram-000NN-of-00016.safetensors`, the same table as the E128 Kaggle Model and candidate 1; `config.json` carries the matching `text_config.engram_compact_kept`.

| | |
|---|---|
| Routed experts | 128 per layer (uniform), 5,120 in all; original ids in `e043_manifest.json`. {SHIFT} |
| Expert selection | contribution scores on model-serving's E065 *training* rollouts (12M image-harness tokens replayed with causal image attention + 12M text-twin tokens); class weights prompt / generated / tool call / image 1/1/10/1 in layers 0-19 and 0/1/10/1 in layers 20-39. The 120 evaluation tasks were never used. |
| Quantization | GPTQ, Humming uint2: gate/up group 128, down group 64, zero points, BF16 scales. Calibration: 1.4M tokens replayed under this mask with causal image attention (half image harness, half text). Experts are stored only in `gptq/layer_LL.safetensors`. |
| Kept as released | attention, router (rows sliced to the kept experts), shared expert, dense layers, vision tower (`vision.*`, `aligner.*`, `image_*`) |
| Dropped | DSpark (`mtp.*`), 256 routed experts per layer, 2/3 of the Engram rows |
| Files | every safetensors file <= 5 GB; {SIZE} bytes |

ARC-AGI-2 eval (120 tasks x 2, image harness, served with the fix, Kaggle key, 115 tasks complete): pass@1 74.9 /
pass@2 84.8, vs 71.2 / 82.9 for the V12 text setup (B32's E128) on the same box. Dev NLL per generated token:
0.8481 (image harness) / 0.8095 (text harness).

## Serving

vLLM DeepSeek-V4.1 SM89/SM120 fork with the patches in `serving/`: `apply_e043_humming.py` + `e043_humming.py`
(2-bit Humming experts from `gptq/`) and `apply_compact_engram.py` (compact Engram, row-chunk loader). Keep the vision
tower (no `--language-model-only`), allow images (`--limit-mm-per-prompt '{"image": 64}'`) and **serve with causal
image attention**: `--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input`.

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

Details: `havu/experiments/E066_dsv41_image_pruning` (section "Fixed image path") and
`havu/experiments/E068_selfhost_image_gap` in `hav4ik/arcprize2026-arcagi2` (private).
