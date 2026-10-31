# DeepSeek-V4.1 (`Image E128` & `Tool E128` 2-Bit) QAT / SFT & Qwen3.8-27B SFT (`sft/`)

This directory implements the end-to-end **Pruned DeepSeek-V4.1 (`Image E128` & `Tool E128` 2-Bit) Multimodal QAT & MoE SFT** and **Qwen3.8-27B Full/LoRA SFT** pipelines for **ARC-AGI-2**:

### A. Pruned DeepSeek-V4.1 (`Image E128` & `Tool E128` 2-Bit) Multimodal QAT & SFT (`v14` & `v12`)
- **Full Experiment Report**: [`../arc_agi_2/reports/dsv41_qat_image_v14_experiment_report.md`](../arc_agi_2/reports/dsv41_qat_image_v14_experiment_report.md) & [`../arc_agi_2/reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md`](../arc_agi_2/reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md)
- **Uploaded Checkpoints**: [`chankhavu/ds-qat-image-v02-ep08`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep08) (`epoch_0p80`, **`+7.14 pp` over Base Image Model** on E058 `text+image` B32) and [`chankhavu/ds-qat-image-v02-ep1-2bit`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep1-2bit) (`epoch_1p00`, **`+3.85 pp` over Base Image Model**, **`-57%` tool-call errors**, **`-27.6%` tokens/attempt**).
- **Core Scripts**:
  - [`dsv41_sft_model.py`](dsv41_sft_model.py): 40-layer `EP=8` PyTorch model, 2-bit Humming `uint2` GPTQ routed experts with trainable `FP32` group scales (`qat_scale`) + `r=32` LoRA, `FP8` block-32 STE shared experts, `64 GiB` Compact Engram, and causal ViT + `3×3` aligner vision tower (`ffn.gate.bias_vl`).
  - [`dsv41_image_tokens.py`](dsv41_image_tokens.py): Exact E058 image-span planner, PNG patch tiler, and `<｜deepseek_image｜>` token expander.
  - [`train_dsv41_sft.py`](train_dsv41_sft.py) & [`launch_dsv41_sft_b200_xm.py`](launch_dsv41_sft_b200_xm.py): `8×B200` / `16×B200` (`DP=2 × EP=8`) QAT & SFT trainer with global gradient clipping, tiered LRs, resumable delta checkpoints, and standalone 2-bit milestone exports.
  - [`eval_dsv41_b32_arc2.py`](eval_dsv41_b32_arc2.py) & [`launch_dsv41_b32_eval_xm.py`](launch_dsv41_b32_eval_xm.py): Distributed `EP=8` KV-cached Candidate B32 evaluation driver (`--grid_format=text+image` or `json`).
  - [`load_dsv41_2bit_example.py`](load_dsv41_2bit_example.py): Reference script for loading standalone 2-bit exports (`export_2bit/epoch_0p80`, `chankhavu/ds-qat-image-v02-ep1-2bit`) or overlaying resumable QAT deltas (`chankhavu/ds-qat-image-v02-ep08`) onto `dsv41_img_e128_2bit_causal`, with text and multimodal greedy decoding.
  - [`audit_and_split_e065_rollouts.py`](audit_and_split_e065_rollouts.py) & [`rank_e065_difficulty_gemini38_flash.py`](rank_e065_difficulty_gemini38_flash.py): E065 multimodal/text rollout audit, ground-truth verification, and difficulty ranking.

---

### B. Qwen3.8-27B ARC-AGI-2 Full SFT & LoRA SFT
1. **3-Gate Data Audit & $D_8$+Color Eval De-Contamination** (`audit_and_prepare_data.py`)
2. **ChatML Tool Rendering & Token-Exact Assistant Loss-Mask Verification** (`data_pipeline.py`)
3. **Custom `@triton.jit` Gated DeltaNet Forward/Backward Kernels (`BT=64`)** (`fast_deltanet.py`)
4. **1×B200 LoRA SFT (`r=64, alpha=128`) Trainer** (`train_sft.py`)
5. **8×B200 Native PyTorch `FSDP2` (`fully_shard`) Full-Parameter (26.9B) Trainer** (`train_full_sft.py`)
6. **Official $128\times 128$ Block-FP8 (`float8_e4m3fn`) Merger & Quantizer for vLLM** (`export_and_quantize.py`)
7. **Interactive HTML Benchmark Report** ([`qwen_sft_lora_report.html`](qwen_sft_lora_report.html))

---

## 1. Headline Evaluation Results (120 Official ARC-AGI-2 Evaluation Tasks)

All runs below are evaluated on the **120 official ARC-AGI-2 evaluation tasks (`167` test pairs)** using the **Comparable SFT-Aligned (`lean2sub`) Harness** ($N=2$ attempts per pair, `334` total attempts, native XML `<function=run_python>` + `<function=submit_answer>`). Under Kaggle's **6-Hour `4×L4` (`96 GB` VRAM) constraint**, the maximum token budget at `~630 tok/s` aggregate throughput is **`13.6M` total tokens**.

| Model / Run | Train Loss | Held-Out Val Loss (PPL) | **Official Kaggle Pass@1** | **Official Kaggle Pass@2** | Median Tokens / Try *(Solved)* | Total Tokens (`167×2` tries) | Fits Kaggle 6h (`4×L4` $\le 13.6\text{M}$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Qwen3.8-27B Baseline** *(Comparable `lean2sub`)* | — | `0.8719` (`2.392`) | `36.39%` (`42.9/118`) | **`38.94%`** (`46.0/118`) | `25,008` *(`16,133`)* | `11.16M` | **YES (`~4.9h`, `82%` cap)** |
| **Best LoRA SFT (`Step 150`, `r=64`)** *(80k Cap)* | `0.5800` | `0.7859` (`2.194`) | **`42.29%`** (`50.3/119`) | **`46.94%`** (`55.9/119`) | `56,423` *(`26,349`)* | `16.84M` | **NO (`~7.4h`, `124%` cap)** |
| **Best Full SFT (`Step 100`, `26.9B`)** *(40k Cap)* | `0.6254` | `0.7689` (`2.157`) *(Step 192: `0.7549`)* | `37.17%` (`42.0/113`) | **`40.76%`** (`46.1/113`) | `39,713` *(`17,578`)* | **`10.35M`** | **YES (`~4.6h`, `76%` cap)** |

* **Highest Raw Accuracy (`+8.00 pp` over baseline)**: **LoRA SFT (`Step 150`)** achieves **`46.94%` Official Pass@2** (`42.29%` Pass@1), outperforming the comparable `lean2sub` baseline (`38.94%`) by **`+8.00 pp`**.
* **Best 6-Hour Kaggle-Compliant SFT Run (`+1.82 pp`, `-0.81M` tokens)**: **Full SFT (`Step 100` at 40k cap)** achieves **`40.76%` Official Pass@2** while generating **`10.35M` total tokens (`~4.6h` on `4×L4`)**—faster and more accurate than the comparable baseline (`38.94%` in `11.16M` tokens / `~4.9h`).
* **Lowest Validation Loss (`0.7549` at `Step 192`)**: Full FSDP2 SFT achieved a **30% larger held-out NLL reduction (`-0.1170` nats, `0.8719 -> 0.7549`)** than LoRA SFT (`-0.0861` nats, `0.8719 -> 0.7858`) in just **75.5 minutes on `8×B200`**.

For full tables, interactive SVG charts, LoRA vs. Full SFT token-length analysis, and a side-by-side comparison with Havu's Tinker API runs (`E028` / `E032`), open **[`qwen_sft_lora_report.html`](qwen_sft_lora_report.html)**.

---

## 2. Script Reference (`sft/` Directory)

| File | Description |
| :--- | :--- |
| **[`audit_and_prepare_data.py`](audit_and_prepare_data.py)** | Audits the `E031–E038` rollout parquet files (`10,104` total rollouts), enforces **3-gate verification** (`correct == True and train_pass == True and answer_source == "submit"`), indexes all 120 ARC-AGI-2 evaluation tasks with exact + $D_8$/color-invariant canonical grid hashes (`canonical_grid_hash`), drops trajectories $> 65,536$ tokens (no mid-turn truncation), and builds the parent-grouped 90/10 train/val dataset (`1,535` train + `64` val trajectories $\le 32,768$ tokens). |
| **[`data_pipeline.py`](data_pipeline.py)** | Renders multi-turn trajectories into Qwen3.8-27B's native ChatML XML tool-calling format (`lean2sub` + `minimal` effort header) and constructs/verifies **token-exact assistant-only loss masks** (`verify_token_loss_mask` asserts 100% token-ID match on `<|im_start|>assistant\n...<|im_end|>`). |
| **[`fast_deltanet.py`](fast_deltanet.py)** | Custom **`@triton.jit` fused forward and backward chunk-recurrent kernels (`BT=64`)** and vectorized cuBLAS triangular solver for Qwen3.8-27B's **48 Gated DeltaNet linear-attention layers**, plus `@torch.compile` fused RMSNorms. Eliminates hundreds of sequential Python kernel launches per token chunk (`5–8×` speedup on linear-attention layers). |
| **[`train_sft.py`](train_sft.py)** | **1×B200 / Multi-GPU DDP LoRA SFT trainer** (`r=64, alpha=128`, `435.2M` trainable params across all 10 attention, Gated DeltaNet, and SwiGLU MLP projections). Uses FP32 master adapter weights, selective gradient checkpointing, and chunked cross-entropy (`chunk_size=2048`). |
| **[`train_full_sft.py`](train_full_sft.py)** | **8×B200 Native PyTorch `FSDP2` (`torch.distributed.fsdp.fully_shard`) full-parameter (`26.90B`) trainer**. Uses `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`, DTensor-aware fused `foreach=True` AdamW (`lr=2e-5`), exact global token-weighted loss reduction across all 8 ranks, preemption-safe distributed checkpoints (`torch.distributed.checkpoint`), and collaborative 8-GPU Block-128 FP8 checkpoint export. Completes 192 steps in **75.5 minutes (`23.6s/step`)**. |
| **[`export_and_quantize.py`](export_and_quantize.py)** | Merges LoRA adapters in FP32 (`W_merged = W_base.float() + (B @ A) * (alpha / r)`) or takes full BF16 weights and quantizes all linear layers to **official $128\times 128$ block-wise FP8 (`float8_e4m3fn` + BF16 `weight_scale_inv`)** for drop-in vLLM TP=2 deployment on `4×L4`. |
| **[`launch_sft_b200_xm.py`](launch_sft_b200_xm.py)** | XManager launcher script for 1×B200 LoRA SFT. |
| **[`launch_full_sft_b200_xm.py`](launch_full_sft_b200_xm.py)** | XManager launcher script for 8×B200 Full FSDP2 SFT (`torchrun --nproc_per_node=8`). |
| **[`sft_test.py`](sft_test.py)** | Comprehensive unit test suite covering ChatML rendering, token-exact loss masking, $D_8$+color contamination hashing, Triton Gated DeltaNet forward/backward parity against PyTorch reference, FSDP2 sharding, and Block-128 FP8 quantization. |
| **[`e031_audit_summary.json`](e031_audit_summary.json)** | Complete JSON audit report of the `E031` dataset split, token percentiles, and contamination checks. |
| **[`qwen_sft_lora_report.html`](qwen_sft_lora_report.html)** | Self-contained HTML report (white background) with all comparison tables, checkpoint progression, SVG charts, training pipeline architecture, and comparison against Havu's GitHub results. |

---

## 3. Data Audit & 3-Gate Verification Summary

### Full Rollout Inventory (`livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036`)

| Parquet File | Rollouts | Unique Tasks | Unique Parent Puzzles | `correct` (Test Exact Match) | `train_pass` (Demo Exact Match) | `train_pass` & NOT `correct` (Demo Overfit) | Contaminated Rows (`2753e76c`, `81c0276b`) | Verified Clean (`correct & train_pass & submit`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e031_arc2_train.parquet` | 1,070 | 994 | 994 | 1,025 (95.8%) | 1,000 (93.5%) | 27 (2.7%) | 0 | **973** |
| `e031_nvarc_b0.parquet` | 1,430 | 1,430 | 590 | 1,006 (70.3%) | 1,163 (81.3%) | **229 (19.7%)** | 0 | **934** |
| **E031 Pilot Total** | **2,500** | **2,424** | **994** | **2,031 (81.2%)** | **2,163 (86.5%)** | **256 (11.8%)** | **0** | **1,907** |
| `e036_nvarc_b1.parquet` | 6,255 | 6,255 | 716 | 4,435 (70.9%) | 5,095 (81.5%) | 983 (19.3%) | 22 | **4,097** |
| `e038_openrouter_deepinfra_b1r.parquet` | 1,246 | 1,246 | 708 | 884 (71.0%) | 1,007 (80.8%) | 186 (18.5%) | 3 | **821** |
| `e037_openrouter_deepinfra.parquet` | 103 | 103 | 89 | 76 (73.8%) | 85 (82.5%) | 14 (16.5%) | 0 | 71 |

### Why 3-Gate Filtering Matters
1. **Eliminating Demo-Overfit Rules (`19.7%` of NVARC)**: On synthetic NVARC puzzles, nearly 1 in 5 trajectories that pass all training demos (`train_pass == True`) fail on the held-out test grid (`correct == False`). Requiring `correct & train_pass & answer_source == "submit"` removes both demo-overfit rules and unverified guesses.
2. **$D_8$ + Color Isomorphism De-Contamination**: Indexing all 120 official evaluation tasks with $D_8$-dihedral and color-permutation invariant canonical grid hashes detected and blocked synthetic families `2753e76c` and `81c0276b` (isomorphic to eval task `89565ca0`).
3. **Zero Parent Leakage (`parent_overlap_count == 0`)**: Grouping by canonical parent puzzle ID (`task_id.split("_")[0]`) ensures no augmented variants of the same parent puzzle cross between the 90% train split (`1,535` trajectories $\le 32\text{k}$ tokens) and 10% validation split (`64` trajectories).

---

## 4. Training Pipeline Architecture: Why Full 27B SFT Finished in 75.5 Minutes

Rather than using Megatron-LM (which lacks optimized kernels for Qwen3.8's hybrid 48 Gated DeltaNet + 16 Full-Attention architecture), we engineered a custom **Native PyTorch `FSDP2` (`fully_shard`) + Custom Triton Gated DeltaNet** pipeline in [`train_full_sft.py`](train_full_sft.py) and [`fast_deltanet.py`](fast_deltanet.py):

1. **Custom `@triton.jit` Fused Chunk-Recurrent Gated DeltaNet Kernels (`BT=64`)**: Replaced HuggingFace's slow sequential Python loop across the 48 linear-attention layers with a vectorized cuBLAS triangular solver $(I + \text{strictLower}(L))^{-1}$ and fused forward/backward Triton state-recurrence kernels (`num_warps=4, num_stages=2`), plus `@torch.compile` fused RMSNorms.
2. **PyTorch `FSDP2` (`fully_shard`) with FP32 Reduce-Scatter**: Each of the 64 decoder layers and the tied `embed_tokens`/`lm_head` is wrapped in `fully_shard` across 8×B200 (`1.43 TB` aggregate HBM3e VRAM), overlapped over NVLink 5 (`1.8 TB/s`/GPU).
3. **Chunked Cross-Entropy (`chunk_size=2048`)**: Computes `lm_head` logits in 2,048-token slices so full `[1, 32768, 248320]` FP32 logits (`32.5 GB`) are never materialized in VRAM.
4. **Parallel 8-GPU Block-128 FP8 Export**: At every checkpoint (`Steps 50, 100, 150, 192`), all 8 B200 ranks collaboratively quantize `safetensors` shards to $128\times 128$ block-FP8 in `< 45 seconds`.
