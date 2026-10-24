# DeepSeek-V4.1-Flash-1017, E128 2-bit GPTQ for tool-integrated ARC-AGI-2 (E043, private)

The same weights as the private HF repo `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`, resharded so that
every safetensors file is at most 5 GB (tensor bytes checked by sha256 against the HF export).

## Model

- **Base:** deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170 (MIT license).
- **Routed experts:** 128 of 384 per backbone layer.
  - Top-6 routing; the shared expert, attention and mHC are unchanged.
  - Kept by a contribution score on generated tokens of 10M tokens of training tool rollouts, with tool-call blocks
    weighted x10.
  - The original expert ids are in `e043_manifest.json`. The router (`ffn.gate.weight` / `bias` / `bias_vl`) is sliced
    to the kept rows.
- **Expert weights:** only low-bit Humming tensors in `gptq/`, with no MXFP4 copies.
  - gate/up are 2-bit with a zero point, group size 128, bf16 scales. down is 2-bit, group size 64.
  - Quantized with E043's per-expert GPTQ. Calibration: 1.4M tokens of tool rollouts replayed under this keep set,
    10% of rows, damping 1.0.
  - Relative output error on the calibration rows, mean over the 40 layers: gate/up 2.5% (RTN 10.9%), down 5.5%
    (RTN 12.7%). `gptq/layer_LL.json` holds the per-layer statistics.
- **Engram:** a compact 64 GiB table, 35% / 32% of rows on layers 1 / 14 (`text_config.engram_compact_kept`).
- **Text only:** DSpark (mtp.*) and vision tensors are removed. Serve with `--language-model-only`.

## File layout: every safetensors file is at most 5 GB

- Backbone shards `model-0000N-of-0000M.safetensors` are at most 5 GB. The GPTQ files in `gptq/` are about 1.24 GB per
  layer.
- Each Engram table (about 32-35 GB as one tensor) is stored as row chunks `layers.{1,14}.engram.embed.weight.rows_<start>`
  in 16 `model-engram-*-of-00016.safetensors` files of at most 4.999 GB. `<start>` is the first compact row of the
  chunk. The scales and `row_map` are whole tensors.

## Serving

Serving needs the vLLM DeepSeek-V4.1 SM89/SM120 fork plus the patches in `serving/`:
- `apply_compact_engram.py`: the compact Engram and the row-chunk loader. `load_weights` copies each chunk's overlap
  into the tensor-parallel rank's slice of the table and checks exact row coverage. Older versions of this patch
  cannot load the chunked Engram; re-running this one upgrades an older patched fork in place.
- `apply_e043_humming.py` with `e043_humming.py`: Humming low-bit experts, router shape (128, 6).

Run each patch script with `<site-packages>/vllm` as its argument. Then set:

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

## Quality

ARC-AGI-2 eval, 120 tasks / 167 pairs, 2 attempts, tool harness `persistent-v6` with sub80s40 + minimal effort header
(80k total output, 40k soft budget, 64k per turn, 40 turns, `--rescue-code`, temperature 1.0). The 120 eval tasks
were used only for scores, never for pruning choices.

| model | pass@1 | pass@2 | correct attempts | mean output tokens | turns per attempt |
|---|---:|---:|---:|---:|---:|
| full model (DeepSeek API) | 74.5% | 84.9% | 247/334 | 27.4K | 13.9 |
| **this checkpoint** | **72.8%** | **82.5%** | 240/334 | 35.2K | 17.1 |

- Parity of the HF export: teacher-forced NLL on 64 dev tool traces 0.79640 (export) vs 0.79652 (runtime masks on the
  full checkpoint); full model 0.7130.
- Known weakness: about 14% of tool calls raise Python errors (identifier typos such as `coms` for `comps`), from
  expert pruning; the model recovers from the error messages at the cost of extra turns.

## 4xL4 (Kaggle) throughput

Real weights, util 0.93, prefix caching, harness with `--token-exact` (exact token history, so re-tokenizing sampled
text never breaks the prefix cache): about 340-350 gen tok/s at 40 streams, 92-96% prefix hits, about 35 attempts/h.
Without `--token-exact` the same run fell to 177 tok/s as the prefix cache broke.

Details: havu/experiments/E043_dsv41_real_pruning in hav4ik/arcprize2026-arcagi2.
