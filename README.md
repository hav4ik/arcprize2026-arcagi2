# ARC-AGI-2: Pruned DeepSeek-V4.1 (`Tool E128` & `Image E128` 2-Bit) QAT / SFT + Qwen3.8-27B SFT & Agentic Python Harnesses

This repository contains the end-to-end training (Multimodal QAT & MoE SFT), 2-bit Humming / Block-FP8 quantization & export, and Havu Candidate B32 evaluation pipelines (`json` text and E058 `text+image`) for **ARC-AGI-2** (`120` official evaluation tasks, `167` test pairs):

1. **`deepseek_sft_qat/` (`google3/.../kaggle/sft/`) — Pruned DeepSeek-V4.1 (`Image E128` & `Tool E128` 2-Bit) Multimodal QAT, MoE SFT & Candidate B32 Evaluation**:
   - **Round-2 Multimodal (`Text + Image`) QAT SFT (`v14` = `round2_v14_16xb200_fast`, `16×B200` `DP=2 × EP=8`)**:
     - Trains `1.887B` routed-expert 2-bit group scales (`qat_scale`) + `3.02B` LoRA (`r=32`) + `1.42B` shared experts + `73.4M` vision aligner (`--train_vision=aligner`) on `6,149` verified E065 multimodal/text trajectories (`e065_img_text_v1`).
     - Updates **`80.2% -> 86.3%` (`1.63B / 1.89B`) of all 2-bit routed-expert group scales** in the exported 2-bit checkpoints (solving `v12`'s `0.00%` 2-bit dead-zone), reducing held-out `val_loss` from `0.7112 -> 0.6715` and `WIC-alnum` typo loss by **`-46.8%` (`0.2252 -> 0.1198`)**.
     - **Apples-to-Apples E058 `text+image` Candidate B32 Evaluation vs. Base Image Model (`dsv41_img_e128_2bit_causal`)**:
       - **`epoch_0p80` ([`chankhavu/ds-qat-image-v02-ep08`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep08))**: **`87.76%` (`43.00/49`) vs Base `80.61%` (`39.50/49`)** [**`+7.14 pp`**], `91.8%` `train_pass` (`+6.1 pp`), `-47%` tool-call errors, `-21.5%` tokens/attempt (`17,385` median).
       - **`epoch_1p00` ([`chankhavu/ds-qat-image-v02-ep1-2bit`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep1-2bit))**: **`86.54%` (`45.00/52`) vs Base `82.69%` (`43.00/52`)** [**`+3.85 pp`**], `92.3%` `train_pass` (`+5.8 pp`), **`-57%` tool-call errors (`15.0% -> 6.4%`)**, **`-27.6%` tokens/attempt (`16,910` median)**.
   - **Round-1 Text-Only MoE SFT (`v12`, `120/120` Tasks Complete)**:
     - **Base `dsv41_e128_2bit` (`120/120` tasks, `N=2`)**: **`71.94%` Official Kaggle Pass@1** (`86.33/120`) / **`80.28%` Official Kaggle Pass@2** (`96.33/120`).
     - **`v12` `epoch_0p25` (`120/120` tasks, `N=2`)**: **`80.69%` Official Kaggle Pass@1** (`96.83/120`, **`+8.75 pp`**) / **`84.31%` Official Kaggle Pass@2** (`101.17/120`, **`+4.03 pp`**).
     - **`v12` `epoch_0p75` (`120/120` tasks, `N=2`)**: **`77.36%` Official Kaggle Pass@1** (`92.83/120`, **`+5.42 pp`**) / **`85.28%` Official Kaggle Pass@2** (`102.33/120`, **`+5.00 pp`**).
2. **`qwen_sft/` (`google3/.../kaggle/sft/`) — Qwen3.8-27B Full-Parameter FSDP2 SFT (`26.9B`) & LoRA SFT (`r=64`)**:
   - **Full SFT Step 100 (`60k` Cap, `N=2`, Block-FP8)**: **`50.0%` Official Kaggle Pass@1** / **`55.0%` Official Kaggle Pass@2** (`66.0/120` tasks).
   - **Full SFT Step 192 (`42k` Fast `<6h` Cap, `N=2`, Block-FP8)**: **`44.0%` Official Kaggle Pass@1** / **`44.9%` Official Kaggle Pass@2** in `10.93M` total tokens (`~4.8h` on `4×L4`).
3. **`qwen_harness/` (`google3/.../kaggle/arc_agi_2/{pilot_1,pilot_2,notebooks}/`) — Havu Candidate B32 Port & Agentic Python Harnesses**:
   - Faithful port of Havu's **Candidate B32** harness (`havu_b32_harness.py`, `havu_auto_repair.py`, `havu_sandbox_zygote.py`, `prompts_and_selector.py`) supporting both DeepSeek (`DSML`, text + E058 `text+image`) and Qwen (`ChatML`) tool calling.

---

## 📊 Technical Reports (`reports/`)

* **[`reports/dsv41_qat_image_v14_experiment_report.md`](reports/dsv41_qat_image_v14_experiment_report.md)** — **NEW: Round-2 Multimodal (`Text + Image`) QAT SFT (`v14`) & Candidate B32 Evaluation Report**:
  - Complete breakdown of `round2_v14_16xb200_fast` (`chankhavu/ds-qat-image-v02-ep08` & `chankhavu/ds-qat-image-v02-ep1-2bit`), `e065_img_text_v1` dataset curation (`6,149` trajectories), trainable 2-bit group scales (`qat_scale`, `86.3%` updated), vision aligner adaptation, 193-step loss progression, E058 `text+image` evaluation tables, scientific conclusions, and model loading/evaluation commands.
* **[`reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md`](reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md)** — **Complete Reference for Pruned DeepSeek-V4.1 (`Tool E128` & `Image E128` 2-Bit) SFT & QAT**:
  - Full model architecture (`40` layers, `MLA+SWA+Compressor+Indexer`, `mHC` 20-iter Sinkhorn, `128` routed experts + `1` shared expert, `64 GiB` Compact Engram on layers 1 & 14, ViT + `3×3` aligner vision tower).
  - Exact quantization formats: `FP8 E4M3` + `UE8M0` power-of-two block-32 scales and 2-bit Humming `uint2` asymmetric Cholesky-GPTQ (`q, s, z` packing with `g128` for `w1/w3` and `g64` for `w2`).
* **[`reports/qat_vs_deepseek_sft_summary_report.md`](reports/qat_vs_deepseek_sft_summary_report.md)** & **[`reports/qat_sft_nemotron_vs_dsv41_report.md`](reports/qat_sft_nemotron_vs_dsv41_report.md)** — Theoretical comparison of NVIDIA Nemotron NVFP4 QAT/QAD vs. 2-bit MoE SFT.
* **[`reports/deepseek_pruning_and_sft_report.md`](reports/deepseek_pruning_and_sft_report.md)** — Deep dive on Havu's `E043` pruning (`384 -> 128` experts with `w=10` tool-block protection, `188.8 -> 64.0 GiB` Engram census) and `uint2` GPTQ (`percdamp = 1.0`).
* **[`reports/dsv41_e128_sft_v12_audit_report.md`](reports/dsv41_e128_sft_v12_audit_report.md)** — Harness fidelity & `v12` checkpoint audit report (`0.0%` routed-expert export delta diagnosis that motivated `v14` QAT).
* **[`reports/havu_b32_vs_pilot2_harness_comparison.md`](reports/havu_b32_vs_pilot2_harness_comparison.md)** & **[`reports/PILOT_2_EXPERIMENTS.md`](reports/PILOT_2_EXPERIMENTS.md)** — Comparison of Havu Candidate B32 vs. `pilot_2` (`h1–h8`).
* **[`reports/qwen_sft_harness_fix_mini_report.md`](reports/qwen_sft_harness_fix_mini_report.md)**, **[`reports/qwen_sft_lora_report.md`](reports/qwen_sft_lora_report.md)**, & **[`reports/ARC_AGI2_SFT_DATA_CATALOG.md`](reports/ARC_AGI2_SFT_DATA_CATALOG.md)** — Qwen3.8-27B Full FSDP2 SFT & LoRA SFT reports and dataset catalog.

---

## 📁 Code Organization (GitHub Branch `arazd` vs. Google3 Paths)

| GitHub Folder (`arazd` branch) | Google3 Package Path | Contents |
| :--- | :--- | :--- |
| **[`deepseek_sft_qat/`](deepseek_sft_qat/)** | `//experimental/users/arazdai/kaggle/sft` | `dsv41_sft_model.py`, `dsv41_image_tokens.py`, `train_dsv41_sft.py`, `eval_dsv41_b32_arc2.py`, `launch_dsv41_sft_b200_xm.py`, `launch_dsv41_b32_eval_xm.py`, `load_dsv41_2bit_example.py`, `audit_and_split_e065_rollouts.py`, `rank_e065_difficulty_gemini38_flash.py`, `inspect_e065_rollouts.py`, `download_hf_to_cns.py`, `inspect_and_download_chankhavu_e128.py`, `data_pipeline.py`, `audit_and_prepare_data.py`, `sft_test.py` |
| **[`qwen_sft/`](qwen_sft/)** | `//experimental/users/arazdai/kaggle/sft` | `train_full_sft.py`, `train_sft.py`, `fast_deltanet.py`, `export_and_quantize.py`, `launch_full_sft_b200_xm.py`, `launch_sft_b200_xm.py`, `evaluate_difficulty_argon.py`, `e031_audit_summary.json` |
| **[`qwen_harness/`](qwen_harness/)** | `//experimental/users/arazdai/kaggle/arc_agi_2` | `pilot_2/` (`havu_b32_harness.py`, `havu_auto_repair.py`, `havu_sandbox_zygote.py`, `prompts_and_selector.py`, `pilot2_eval.py`, `sandbox_verifier.py`), `pilot_1/` (`setB/`), `notebooks/`, `vllm_cuda_shim.py`, `vllm_fla_triton_ops.py` |
| **[`inspect_tools/`](inspect_tools/)** | `//experimental/users/arazdai/inspect_tools` | `safetensors_param_census.py` (Safetensors parameter census & byte-size inspector) |
| **[`reports/`](reports/)** | `//experimental/users/arazdai/kaggle/arc_agi_2/reports` | `dsv41_qat_image_v14_experiment_report.md`, `DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md` + DeepSeek & Qwen technical reports |

---

## 🚀 1. How to Load & Evaluate the Latest QAT Image SFT Models (`load_dsv41_2bit_example.py`)

[`load_dsv41_2bit_example.py`](../sft/load_dsv41_2bit_example.py) supports both **standalone exported 2-bit checkpoints** (e.g. `export_2bit/epoch_0p80`, `export_2bit/epoch_1p00`, or HuggingFace [`chankhavu/ds-qat-image-v02-ep1-2bit`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep1-2bit)) and **resumable QAT delta checkpoints** (e.g. `step-00154-epoch_0p80` or HuggingFace [`chankhavu/ds-qat-image-v02-ep08`](https://huggingface.co/chankhavu/ds-qat-image-v02-ep08) overlaid onto `dsv41_img_e128_2bit_causal`), plus both text-only (`greedy_generate`) and E058 multimodal `text+image` (`greedy_generate_with_images`) decoding:

```bash
# 1. Load a standalone exported 2-bit image checkpoint (e.g. epoch_0p80 or chankhavu/ds-qat-image-v02-ep1-2bit):
torchrun --nproc_per_node=8 load_dsv41_2bit_example.py \
  --checkpoint_2bit_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/export_2bit/epoch_0p80 \
  --engram_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --enable_qat_eval=true

# 2. Load the base image model + overlay the resumable QAT delta (chankhavu/ds-qat-image-v02-ep08)
#    and optionally export it to a standalone 2-bit directory:
torchrun --nproc_per_node=8 load_dsv41_2bit_example.py \
  --base_image_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal \
  --resumable_ckpt_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/step-00154-epoch_0p80 \
  --engram_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --export_2bit_out_dir=/tmp/exported_ds_qat_image_v02_ep08_2bit
```

---

## 🎯 2. How to Run ARC-AGI-2 Evaluation with Havu's Best Harness (`Candidate B32`)

[`eval_dsv41_b32_arc2.py`](../sft/eval_dsv41_b32_arc2.py) loads any 2-bit checkpoint onto `8×B200` (`EP=8`), runs a batched KV-cached decode loop (`24` concurrent streams per 8-GPU shard, `max_model_len = 163,840`), and drives [`havu_b32_harness.py`](pilot_2/havu_b32_harness.py) (`N=2` attempts per task, `80k` hard / `40k` gentle budget, multi-test scaling, AST/OSA ≤ 2 auto-repair, 4-path `train_check`, and `ranked_answers`) in either `--grid_format=text+image --image_cell=0` (E058 multimodal) or `--grid_format=json` (text-only) mode.

```bash
# Launch 4-shard E058 Multimodal (text+image) Candidate B32 Evaluation on XManager:
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_b32_eval_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_qat_img_v14_b32_eval \
  --sft_output_root=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1 \
  --eval_output_root=/cns/lu-d/home/arazdai/kaggle/sft/evals/v14_e065_img_b32_eval \
  --base_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal \
  --base_engram_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --eval_base_model=true \
  --milestones=0.20,0.40,0.60,0.80,1.00 \
  --grid_format=text+image \
  --image_cell=0 \
  --num_shards=4 \
  --num_attempts=2
```

---

## 🔥 3. How to Launch Round-2 Multimodal QAT SFT (`16×B200`, `DP=2 × EP=8`)

```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_sft_b200_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_img_e128_qat_sft_16xb200_v14 \
  --model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal \
  --engram_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --data_dir=/cns/lu-d/home/arazdai/kaggle/sft/data/e065_img_text_v1 \
  --output_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1 \
  --num_nodes=2 \
  --train_moe_only=true \
  --train_vision=aligner \
  --enable_qat=true \
  --qat_trainable_expert_scales=true \
  --global_grad_clip=true \
  --load_compact_engram=true \
  --routed_expert_lora_r=32 \
  --routed_expert_lora_alpha=64 \
  --learning_rate=1e-5 \
  --num_epochs=1.0 \
  --grad_accum_steps=2 \
  --max_seq_len=32768 \
  --export_2bit_milestones=true \
  --milestone_fractions=0.20,0.40,0.60,0.80,1.00 \
  --auto_resume=true
```

### Run Unit Tests (`sft_test` & `havu_b32_harness_test`)
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
  //experimental/users/arazdai/kaggle/sft:sft_test \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:havu_b32_harness_test \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_test
```
