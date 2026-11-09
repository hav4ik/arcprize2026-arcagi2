---
license: mit
---
# DeepSeek-V4.1-Flash-1017 · ARC tool use with images · per-layer 72-152 experts (avg 120), 2-bit (E066)

DeepSeek-V4.1-Flash-1017 pruned to a **different number of routed experts per layer** (72 to 152 of 384, 120 on
average) and quantized to 2 bits, for ARC-AGI-2 tool use on Kaggle 4xL4 with the image harness (E058) and its text
twin. The vision tower is kept.

| | |
|---|---|
| Routed experts | 4,800 in all (as uniform 120). Per layer, in steps of 16 (6 distinct counts): `72,72,72,152,152,152,152,72,152,136,152,152,152,152,104,152,152,152,104,152,120,104,120,152,88,72,72,72,72,104,104,72,88,104,88,152,152,152,152,152`, also in `text_config.n_routed_experts_per_layer` (`n_routed_experts` = 152, the maximum) |
| Allocation | per-layer sensitivity sweep (dev NLL with each layer cut to 64 experts) scaled into a share curve, then greedy in steps of 16 within floor 72 / cap 152 |
| Expert selection | as candidate 1: contribution scores on model-serving's E065 *training* rollouts, image harness and text twin; class weights 1/1/10/1 (layers 0-19), 0/1/10/1 (20-39). The 120 evaluation tasks were never used. |
| Quantization | GPTQ, Humming uint2: gate/up group 128, down group 64, zero points, BF16 scales; calibration 1.4M tokens replayed under this mask (half image, half text). `gptq/layer_LL.safetensors` holds e0..e{E_l-1}, compact ids = rank among the kept original ids (`e043_manifest.json`). |
| Kept as released | attention, router (rows sliced per layer), shared expert, dense layers, vision tower |
| Dropped | DSpark (`mtp.*`), the other routed experts, 2/3 of the Engram rows |
| Engram | 64 GiB tool table, the same 16 row-chunk files as the E112 repo and the Kaggle E128 model |
| Files | every safetensors file <= 5 GB; {SIZE} bytes |

**Dev NLL per generated token** (64 held-out training rollouts, family-disjoint split; lower is better):

| model | image harness | text harness |
|---|---|---|
| full model | 0.7522 | 0.7190 |
| E128 candidate 1 keep set, before quantization | 0.8166 | 0.7766 |
| this keep set, before quantization (paired vs candidate 1) | -0.0115 | -0.0094 |
| E128 candidate 1, 2-bit | 0.8482 | 0.8068 |
| **this model (2-bit)** | **{Q_IMG}** | **{Q_TXT}** |

Kaggle 4xL4 KV pool with vision (dsv4-infer, dummy weights of this exact layout): **3.38M tokens** vs 2.93M for
uniform 128; decode speed unchanged.

Checks: every GPTQ tensor and router row equals its source byte-for-byte (`check_lowbit_export.py`). The 2-bit numbers
above come from runtime masks on the full checkpoint (the E066 eval fork cannot load per-layer exports); serving parity
of this export on the dynamic-count fork: pending (dsv4-infer).

**ARC-AGI-2 evaluation** (120 tasks, 2 attempts): pending.

## Serving

Needs the dynamic-count MoE patch (per-layer gate / bias / bias_vl / FusedMoE sizes from
`n_routed_experts_per_layer`, router whitelist 6..512): `havu/experiments/E062_manh_vllm_fork/port/04_per_layer_experts.patch`
in `hav4ik/arcprize2026-arcagi2`, on Manh's DeepSeek-V4.1 fork. Plus the patches in `serving/` (Humming experts from
`gptq/`, compact Engram row-chunk loader), the vision tower kept, and:

    export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
    export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
    export E043_GPTQ_DIR=<model_dir>/gptq

Details: `havu/experiments/E066_dsv41_image_pruning` in `hav4ik/arcprize2026-arcagi2` (private).
