# Serving the static active-expert schedules

## Tested experimental stack

The included `apply_active_topk_e043.py` targets the E043 vision11 fork with files
`vllm/models/deepseek_v4_1/nvidia/model.py` and `router.py`. It extends the E128
router allowlist and reads one integer top-k per layer from `E074_TOPK_FILE`
before constructing FusedMoE. It is idempotent and refuses unsupported layouts.
The patch and 36 router reference cases passed on the experimental stack;
seed14's real model ran image/text NLL and generated tool diagnostics there.

After installing that stack and the existing compact-Engram/Humming support:

```bash
python "$MODEL_DIR/serving/apply_active_topk_e043.py" "$VLLM_PACKAGE_DIR"
export E074_TOPK_FILE="$MODEL_DIR/active_experts.json"
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export DSV41_EXPERT_HUMMING='{"quant_method":"humming","weight_dtype":"uint2","weight_scale_dtype":"bfloat16","group_size":128,"has_zero_point":true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size":64}'
```

Use the normal E043 TP4 serving command with prefix caching, compact Engram CPU
offload, native FP8 KV, the DeepSeek V4.1 tokenizer/tool parsers, and
`--hf-overrides '{"is_mm_prefix_lm":false}' --disable-chunked-mm-input`.
The E076 calibration server uses max-model-len 65536; full ARC uses 163840.
Check startup logs for all 40 `E074 layer=... top_k=...` lines and match them to
the selected JSON. The active-top-k file is the runtime source on this stack.

For the evaluated mean-five alternative, change E074_TOPK_FILE to
`$MODEL_DIR/schedules/e074_mean5_14p58B.json` and restart the engine. Its active
schedule overrides the preview metadata on this experimental stack; report its
separate name and evidence. Baseline k6 is in `schedules/baseline_k6.json`.

## Manh's current fork: port required, not yet GPU validated

Read-only audit of `g22df163d2d`, Humming 0.1.18:

1. `models/deepseek_v41/nvidia/model.py`: read and validate the per-layer list,
   then pass the selected layer's value as `n_activated_experts` to the base MoE.
2. `model_executor/layers/fused_moe/router/dsv4_topk.py`: replace fixed-six
   eligibility, loop bound, output stride/mask, allocation and warmup shapes with
   the requested compile-time TOP_K (3..6 for E076).
3. `fused_topk_bias_router.py`: forward topk to the router call.

Humming already consumes topk_ids/topk_weights through the modular interface.
The GB300-only opt-in latency_specialized path has additional fixed-six kernels;
it is outside the L4/RTX PRO 6000 serving path and must not be enabled here.

Before claiming compatibility, run text/image router reference cases including
ties and padding, an unchanged-k6 check, real mixed-k Humming prefill/decode with
CUDA graphs, and long-context prefix-cached tool traffic. A passing model load
alone is insufficient. Full Kaggle submission qualification remains separate.
