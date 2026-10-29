---
license: mit
---
# DeepSeek-V4.1-Flash-1017 · ARC tool use with images · E128 2-bit (E066 candidate 1)

DeepSeek-V4.1-Flash-1017 pruned to 128 of 384 routed experts per layer and quantized to 2 bits, for ARC-AGI-2 tool
use on Kaggle 4xL4 with the image harness (E058: the E046 tool harness plus one image per grid) and its text twin.
The vision tower is kept.

| | |
|---|---|
| Routed experts | 128 per layer (uniform), 5,120 in all; original ids in `e043_manifest.json` |
| Expert selection | contribution scores on model-serving's E065 *training* rollouts, image harness and text twin; class weights prompt / generated / tool call / image 1/1/10/1 in layers 0-19 and 0/1/10/1 in layers 20-39. The 120 evaluation tasks were never used. |
| Quantization | GPTQ, Humming uint2: gate/up group 128, down group 64, zero points, BF16 scales. Calibration: 1.4M tokens replayed under this mask (half image harness, half text). Experts are stored only in `gptq/layer_LL.safetensors`. |
| Kept as released | attention, router (rows sliced to the kept experts), shared expert, dense layers, vision tower (`vision.*`, `aligner.*`, `image_*`) |
| Dropped | DSpark (`mtp.*`), 256 routed experts per layer, 2/3 of the Engram rows |
| Engram | 64 GiB tool table (top-frequency rows): the same 16 row-chunk files as `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit` and the Kaggle E128 model |
| Files | every safetensors file <= 5 GB; {SIZE} bytes |

**Dev NLL per generated token** (64 held-out training rollouts, family-disjoint dev split; lower is better):

| model | image harness | text harness |
|---|---|---|
| full model | 0.7522 | 0.7190 |
| this keep set, before quantization | 0.8166 (+8.6%) | 0.7766 (+8.0%) |
| **this model (2-bit)** | **{Q_IMG}** | **{Q_TXT}** |
| shipped tool E128 keep set, before quantization (reference) | 0.8192 | 0.7786 |

Serving parity: this export served from its own files gives dev NLL {PARITY} (the line above is the same model with
runtime masks on the full checkpoint). Kaggle 4xL4 KV pool with vision (dsv4-infer, dummy weights): 2.93M tokens.

**ARC-AGI-2 evaluation** (120 tasks, 2 attempts): pending.

## Serving

vLLM DeepSeek-V4.1 SM89/SM120 fork with the patches in `serving/`: `apply_e043_humming.py` + `e043_humming.py`
(2-bit Humming experts from `gptq/`) and `apply_compact_engram.py` (compact Engram, row-chunk loader). Keep the vision
tower (no `--language-model-only`) and allow images (`--limit-mm-per-prompt '{"image": 64}'`).

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

Details: `havu/experiments/E066_dsv41_image_pruning` in `hav4ik/arcprize2026-arcagi2` (private).
