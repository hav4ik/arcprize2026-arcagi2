# Quantization-Aware SFT: NVIDIA Nemotron (NVFP4) vs. Pruned DeepSeek-V4.1 (Tool E128)

**Executive Summary:** When post-training (SFT/RL) models deployed at ultra-low bitwidths (`2-bit` to `4-bit`), standard *"De-quantize to BF16 → SFT in BF16 → Post-Training Re-Quantize (PTQ)"* loses a significant fraction of SFT gains at export time. NVIDIA's **Nemotron NVFP4** recipe—spanning **NVFP4 Pretraining** ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)) and **Quantization-Aware Distillation (QAD)** ([arXiv:2601.20088](https://arxiv.org/abs/2601.20088))—shows that preserving reasoning and MoE routing accuracy requires:

1. **Simulated Quantization (Fake-Quant + STE) in the Forward Pass** during training.
2. **FP32 Master Weights** so small SFT updates (`1e-6` to `1e-5`) do not underflow `BF16` mantissa precision before crossing quantization thresholds.
3. **Quantization-Aware Distillation (QAD) via Forward KL (`T = 1.0`)** from a frozen high-precision teacher to stabilize MoE router inputs and multi-step reasoning chains.
4. **Activation-Aware Scale Preservation** rather than data-free Frobenius/L2 re-packing.

---

## 1. How NVIDIA / Nemotron Does NVFP4-Aware Pretraining & SFT

* **NVFP4 Format (`W4A4`):** 16-element micro-blocks in 4-bit `E2M1` (`±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`) with local `FP8 (E4M3)` micro-block scales + global per-tensor `FP32` scale. Both weights and activations are quantized in the forward pass.
* **NVFP4 Pretraining Guardrails ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)):**
  * **Selective Layer Precision (~15% kept in BF16/FP32):** Embeddings, LM head, RMSNorms, attention QKV/softmax, **MoE router gates**, and the **final ~15% of transformer blocks** stay in high precision.
  * **2D Block Scaling (`16×16`) for Weights vs. 1D (`1×16`) for Activations:** Ensures weight representations match between forward (`W`) and backward (`W^T`) passes.
  * **Random Hadamard Transforms (`16×16` RHT):** Flattens outlier spikes on weight-gradient GEMM inputs.
  * **FP32 Master Weights + Stochastic Rounding on Gradients:** Prevents update stalling/bias on narrow bitwidths.
* **Nemotron Post-Training / SFT Recipe (QAT & QAD, [arXiv:2601.20088](https://arxiv.org/abs/2601.20088)):**
  * **Fake-Quantization (STE) Inside `forward()`:** Linear layers apply `W_hat = Dequant(Quant(W_master))` with Straight-Through Estimator gradients so the optimizer places weights directly onto valid low-bit lattice points.
  * **Quantization-Aware Distillation (QAD) over Hard-Label CE:** Standard hard-label cross-entropy under 2-bit/4-bit quantization causes high gradient variance and shifts hidden states entering downstream **MoE routers**. QAD freezes the high-precision teacher and minimizes **Forward KL Divergence at `T = 1.0`** (`KL(P_teacher || P_quant_student)`), recovering near-lossless accuracy on *Nemotron 3 Nano 30B-A3B MoE*.

---

## 2. Summary of Our Current SFT Protocol on Pruned DeepSeek-V4.1 (Tool E128)

* **Base Checkpoint Format (`dsv41_e128_2bit`):**
  * **Backbone (`model-00001/00002.safetensors`):** `float8_e4m3fn` weights + `UE8M0` power-of-two block-32 scales (`weight_scale_inv`) for attention (`wq_a, wq_b, wkv_a, wkv_b, wo`) and `shared_experts`, plus `BF16` norms, mHC, embeddings, LM head, and 128-expert router gates (`4.91B` params).
  * **128 Routed Experts/Layer (`gptq/layer_00..39.safetensors`):** **2-bit Humming (`uint2`)** asymmetric GPTQ format `(q, s, z)` with `group=128` (`w1, w3`) and `group=64` (`w2`), originally calibrated with activation-aware GPTQ Hessians.
* **Step-by-Step Current Pipeline (`train_dsv41_sft.py` & `dsv41_sft_model.py`):**
  1. **De-quantize to BF16 at Load:** Unpacks all FP8 block-32 backbone weights and 2-bit `(q, s, z)` experts into dense `BF16` tensors across 8× B200 GPUs; discards original `(q, s, z)` and `UE8M0` scales.
  2. **Train MoE Only (`train_moe_only=True`):** Freezes attention, routers, mHC, norms, embed, and head. Trains `shared_experts` densely in `BF16` + 128 `routed_experts` via `LoRALinear (r=32, alpha=64)` in `BF16` at `lr=8e-6` / `1e-5`. (Compact Engram lookup on layers 1 & 14 is currently bypassed during SFT).
  3. **Unquantized Forward & Validation:** Runs forward/backward and `val_loss` in continuous `BF16` with hard-label token-weighted CE (`WIC=10.0, submit=5.0, tool=2.0`).
  4. **Data-Free PTQ Re-Quantization at Milestone Export:** Merges LoRA into routed experts (`W_base + B @ A * (alpha/r)`) and runs a 16-point min-max L2 grid search over `z in {0..3}` and `shrink in {1.0, 0.95, 0.90, 0.85}`; re-quantizes `shared_experts` using IEEE-754 power-of-two ceiling (`exp_ceil`) `UE8M0` block-32 scales.

---

## 3. Key Differences & Quantization Vulnerabilities

| Dimension | NVIDIA Nemotron (NVFP4 QAT/QAD) | Our DSV4.1 E128 (2-Bit/FP8) SFT | Vulnerability in Our Setup |
| :--- | :--- | :--- | :--- |
| **1. Forward Pass** | Fake-Quant (STE) active during training | Continuous BF16 (quantized only at export) | Small LoRA updates get rounded back to zero or trigger discrete jumps at export. |
| **2. 2-Bit Expert Export** | Activation-calibrated scales preserved/trained | Data-free L2 min-max grid search on `W + LoRA` | Flipping `(s, z)` via unweighted L2 search destroys Havu's GPTQ Hessian error compensation across the 64/128-weight group. |
| **3. Block Scales (`UE8M0`)** | Smooth FP8 micro-scales visible in forward pass | Power-of-2 ceiling (`exp_ceil`) applied at export | Crossing a `2^k` boundary at export halves FP8 precision for all 32 weights in that block. |
| **4. Master Weights** | FP32 master weights | BF16 parameters (`lr=8e-6`) | Updates (`~1e-6`) fall below BF16 ULP (`~8e-4` for `\|w\|~0.1`), causing update underflow. |
| **5. Loss & Eval** | Forward KL Distillation (QAD) + quantized eval | Hard-label CE + BF16 validation | `val_loss` improves in BF16 while exported 2-bit checkpoint degrades and shifts frozen MoE routing. |

---

## 4. Prioritized Improvements to Implement

1. **Add STE Fake-Quantization in `forward()` & Preserve GPTQ `(q, z)` Structure:**
   * Wrap `shared_experts` weights in an STE fake-quantizer calling `quantize_fp8_block32_torch` + `dequantize_fp8_block32_torch` so training sees exact `UE8M0` power-of-two scale ceilings.
   * For **2-bit Humming routed experts**: instead of LoRA + destructive 2-bit L2 re-packing, **freeze Havu's GPTQ integer codes `q` and zero-points `z` and directly train the per-group scales `s`** (or apply STE with locked `z_orig`). Training `s` (`~360M` parameters across 40 layers) preserves 100% of GPTQ Hessian compensation and exports to `gptq/layer_XX.safetensors` with **zero re-packing error**.
2. **Keep Trainable Master Weights in FP32:**
   * Store trainable `shared_experts` and `lora_A/lora_B` (or group scales `s`) in `torch.float32` and cast to `bfloat16` in `forward()`. Adds `< 2.5 GiB/GPU` on B200 and eliminates BF16 underflow at `lr=8e-6`.
3. **Evaluate Validation Loss on Quantized Weights (`val_quant_loss`):**
   * Run `evaluate_dsv41_validation_loss` with fake-quantized weights enabled so checkpoint selection reflects true post-export 2-bit/FP8 perplexity.
4. **Add Forward KL Regularization (Nemotron QAD Objective):**
   * Blend weighted CE with a Forward KL penalty (`T = 1.0`) against the frozen base E128 model's logits so quantized expert updates do not perturb downstream frozen 128-expert routers.
5. **Enable Compact Engram Lookup (Layers 1 & 14) During SFT & Eval:**
   * Load the compact Engram tables during SFT and B32 evaluation so layers 1 and 14 match inference-time residual distributions.
