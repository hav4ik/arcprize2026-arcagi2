# DeepSeek-V4.1-Flash-1017, E112 2-bit GPTQ for tool-integrated ARC-AGI-2 (E043, private)

Same recipe as `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`, with 112 instead of 128 routed experts per
layer.

## Model

- **Base:** deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170.
- **Routed experts:** 112 of 384 per backbone layer.
  - Top-6 routing; the shared expert, attention and mHC are unchanged.
  - Kept by a contribution score on generated tokens of 10M tokens of training tool rollouts, with tool-call blocks
    weighted x10 (94.9% of the tool-block contribution is kept).
  - The kept experts are a subset of the tool E128 set.
  - The original expert ids are in `e043_manifest.json`. The router (`ffn.gate.weight` / `bias` / `bias_vl`) is sliced
    to the kept rows.
- **Expert weights:** only low-bit Humming tensors in `gptq/`, with no MXFP4 copies.
  - gate/up are 2-bit with a zero point, group size 128, bf16 scales. down is 2-bit, group size 64.
  - Quantized with E043's per-expert GPTQ (`gptq_experts.py`). Calibration: the tool E128 model's calibration
    rollouts (1.4M tokens of tool rollouts), replayed under this keep set, 10% of rows, damping 1.0.
  - Relative output error on the calibration rows, mean over the 40 layers: gate/up 2.64% (RTN 10.9%), down 5.64%
    (RTN 12.7%). The worst layer is at 3.6% / 7.2%. Every layer decodes bit-exactly through Humming.
  - `gptq/layer_LL.json` holds the per-layer statistics.
- **Engram:** the compact 64 GiB tool table of the E128 model, 35% / 32% of rows on layers 1 / 14
  (`text_config.engram_compact_kept`).
- **Text only:** DSpark (mtp.*) and vision tensors are removed. Serve with `--language-model-only`.
- **Size:** 124,890,565,440 bytes.

## File layout: every safetensors file is at most 5 GB

- Backbone shards are capped at 4.9 GB. The GPTQ files are ~1.1 GB per layer.
- Each Engram table (~32-35 GB as one tensor) is stored as row chunks:
  `layers.{1,14}.engram.embed.weight.rows_<start>`, in 16 `model-engram-*` files of at most 4.999 GB.
  - `<start>` is the first compact row of the chunk.
  - Scales and `row_map` are whole tensors (1-1.5 GB).
- The chunks are byte-identical slices of the E128 model's Engram tables (sha256-checked when they were written).

## Serving

Serving needs the vLLM DeepSeek-V4.1 SM89/SM120 fork plus the patches in `serving/`:
- `apply_compact_engram.py` (this version): compact Engram, and the row-chunk loader. `load_weights` copies each
  chunk's overlap into the TP rank's slice of the table and checks that every row was covered. Older versions of the
  patch cannot load this checkpoint's Engram.
- `apply_e043_humming.py` with `e043_humming.py`: Humming low-bit experts, router shape (112, 6).

Run each patch script with `<site-packages>/vllm` as its argument; both are idempotent. Then set:

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

## Checks

- Content, from the files: every GPTQ tensor equals the source GPTQ tensor at its original expert id, and the router
  rows equal the original checkpoint's rows at the kept ids.
- **Dev NLL** with runtime masks on the full checkpoint (64 family-disjoint dev tool traces, generated tokens): **0.8233 (+15.5% vs the full model)**; tool-block NLL 0.158, tool-block top-1 94.5%.
  - For reference: full model 0.7130; tool E128 2-bit 0.7965 (+11.7%); tool E96 2/3-bit 0.8318 (+16.7%).
- **Serving parity of this file layout** (row-chunked Engram through the `serving/` loader, vLLM TP4, 2026-10-24):
  dev NLL 0.82336 vs 0.82330 with runtime masks on the full checkpoint; tool-block NLL 0.1573 vs 0.1577.

## Quality

ARC-AGI-2 eval, 120 tasks / 167 pairs, tool harness `persistent-v6` with sub80s40 + minimal effort header (80k total
output, 40k soft budget, 64k per turn, 40 turns, `--rescue-code`, `--token-exact`, temperature 1.0). The 120 eval
tasks were used only for scores, never for pruning choices.

| model | pass@1 | pass@2 | correct attempts | Python-error tool calls |
|---|---:|---:|---:|---:|
| full model (DeepSeek API) | 74.5% | 84.9% (142/167) | 247/334 | 2.3% |
| tool E128 2-bit | 72.8% | 82.5% (136/167) | 240/334 | 14% |
| **this checkpoint, single-process runs** (attempt 0 + rerun r2) | **70.2%** | **78.1% (128/167)** | 230/334 | 29% |
| this checkpoint, official file (attempt 0 + attempt 1) | 64.8% | 75.6% (124/167) | 211/334 | 29% |

- Single attempts: 116 and 114 of 167 when the eval process ran alone; 95 and 101 when the process was started next
  to another running E112 eval on the same two servers (attempt 1, rerun r3). Every recorded setting was identical;
  the cause is not found (not the masks, the code version, the cache salt, the sandbox, or measurable load numerics).
  The single-process numbers are the reliable ones.
- About 29% of tool calls raise Python errors (identifier typos such as `coms` for `comps`), about twice the E128
  model's rate. The model recovers from them, at about 3 extra tool calls per attempt, and hits the 40-turn cap more
  often on hard tasks.

## 4xL4 footprint (estimate from the shapes, not measured)

At util 0.93: weights ~15.4 GiB per GPU (E128: 16.87), KV ~3.5 GiB, about 2.5M tokens (E128: 1.47M). That is about
the same footprint as the E96 2/3-bit model.

Details: havu/experiments/E043_dsv41_real_pruning in hav4ik/arcprize2026-arcagi2.
