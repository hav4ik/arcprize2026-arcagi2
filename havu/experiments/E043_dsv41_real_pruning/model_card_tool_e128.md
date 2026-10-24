# DeepSeek-V4.1-Flash-1017, E128 2-bit GPTQ for tool-integrated ARC-AGI-2 (E043 tool round, private)

## Model

- **Base:** deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170.
- **Routed experts:** 128 of 384 per backbone layer.
  - Top-6 routing; the shared expert, attention and mHC are unchanged.
  - Kept by a contribution score on generated tokens of 10M tokens of training tool rollouts, with tool-call blocks
    weighted x10.
  - The original expert ids are in `e043_manifest.json`. The router (`ffn.gate.weight` / `bias` / `bias_vl`) is sliced
    to the kept rows.
- **Expert weights:** only low-bit Humming tensors in `gptq/`, with no MXFP4 copies.
  - gate/up are 2-bit with a zero point, group size 128, bf16 scales. down is 2-bit, group size 64.
  - Quantized with E043's per-expert GPTQ (`gptq_experts.py`). Calibration: 1.4M tokens of tool rollouts replayed
    under this keep set, 10% of rows, damping 1.0.
  - Relative output error on the calibration rows, mean over the 40 layers: gate/up 2.5% (RTN 10.9%), down 5.5%
    (RTN 12.7%). The worst layer is at 3.5% / 7.1%.
  - `gptq/layer_LL.json` holds the per-layer error statistics.
- **Engram:** a compact 64 GiB table, 35% / 32% of rows on layers 1 / 14.
  - Rows chosen by top lookup frequency on the tool calibration traffic.
  - Stored in `model-engram-*.safetensors`; `text_config.engram_compact_kept` is set.
  - On this 2-bit model, dev NLL is 0.79679 with 64 GiB, 0.79647 with 112 GiB and 0.79650 with the full table.
- **Text only:** DSpark (mtp.*) and vision tensors are removed. Serve with `--language-model-only`.
- **Size:** 131,090,957,662 bytes.

## Serving

Serving needs the vLLM DeepSeek-V4.1 SM89/SM120 fork plus E043's patches: `apply_e043_patches.py`,
`apply_compact_engram.py`, and `apply_e043_humming.py`, which adds the router shape (128, 6). Set:

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

## Quality

**Harness:** ARC-AGI-2 eval, 120 tasks / 167 pairs, 2 attempts. Tool harness `persistent-v6` with sub80s40 + minimal
effort header: 80k total output, 40k soft budget, 64k per turn, 40 turns, `--rescue-code`, temperature 1.0. The 120
eval tasks were used only for these scores, never for pruning choices.

| model | pass@1 | pass@2 | correct attempts | mean output tokens | turns per attempt |
|---|---:|---:|---:|---:|---:|
| full model (DeepSeek API) | 74.5% | 84.9% | 247/334 | 27.4K | 13.9 |
| **this checkpoint** | **72.8%** | **82.5%** | 240/334 | 35.2K | 17.1 |

pass@1 and pass@2 are task-weighted.

- Pairs: 128 solved by both, 14 by the full model only, 8 by this model only.
- 27 attempts hit the 80k output budget, against 2 for the full model.
- **Parity:** teacher-forced NLL on 64 family-disjoint dev tool traces (generated tokens) is 0.79640 for this export,
  0.79652 for runtime masks on the full checkpoint, and 0.7130 for the full model (+11.7%). Tool-block NLL is 0.149
  here and 0.118 for the full model.
- **Known weakness:** 14% of tool calls raise Python errors (full model: 2.3%), mostly identifier typos such as `coms`
  for `comps` or `scpy` for `scipy`. They come from expert pruning, not the 2-bit quantization, and min_p / top_p
  sampling does not remove them. The model recovers from the error messages, at the cost of extra turns.

## 4xL4 throughput

E042, dummy weights in this schema, util 0.93, prefix caching, replaying the full model's tool turn shapes: 340.5 gen
tok/s at 48 streams, 96.1% prefix hits, 0 preemptions. This model's own turns are about 15% longer, so about 40
streams fit.

Details: havu/experiments/E043_dsv41_real_pruning in hav4ik/arcprize2026-arcagi2.
