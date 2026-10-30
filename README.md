# ARC-AGI-2: Pruned DeepSeek-V4.1 (`Tool E128` 2-Bit) SFT & QAT + Qwen3.8-27B SFT & Agentic Python Harnesses

This repository contains the end-to-end training (SFT & QAT), 2-bit Humming / Block-FP8 quantization & export, and Havu Candidate B32 evaluation pipelines for **ARC-AGI-2** (`120` official evaluation tasks, `167` test pairs):

1. **`deepseek_sft_qat/` (`google3/.../kaggle/sft/`) — Pruned DeepSeek-V4.1 (`Tool E128` 2-Bit) SFT, QAT & Candidate B32 Evaluation**:
   - **Base `dsv41_e128_2bit` (`120/120` tasks, `N=2`)**: **`71.9%` Official Kaggle Pass@1** (`86.33/120`) / **`80.3%` Official Kaggle Pass@2** (`96.33/120`), median `36,506` output tokens/attempt.
   - **MoE-Only SFT `v12` (`epoch_0p25` = Step 28, `116/120` tasks done)**: **`81.3%` Official Kaggle Pass@1 on done tasks (`94.33/120 = 78.6%` lower bound across all 120)** / **`85.1%` Official Kaggle Pass@2 on done tasks (`98.67/120 = 82.2%` lower bound across all 120 — already beating Base's 120-task final score with 4 tasks still running)**, median `33,439` output tokens/attempt.
   - **Validation Loss (`114` steps = `1.00` epoch)**: Held-out validation loss drops cleanly from `0.6768 -> 0.6599`, and word-internal alphanumeric continuation (`WIC-alnum`) loss drops by **`-35.5%` (`0.2155 -> 0.1389`)**.
2. **`qwen_sft/` (`google3/.../kaggle/sft/`) — Qwen3.8-27B Full-Parameter FSDP2 SFT (`26.9B`) & LoRA SFT (`r=64`)**:
   - **Full SFT Step 100 (`60k` Cap, `N=2`, Block-FP8)**: **`50.0%` Official Kaggle Pass@1** / **`55.0%` Official Kaggle Pass@2** (`66.0/120` tasks).
   - **Full SFT Step 192 (`42k` Fast `<6h` Cap, `N=2`, Block-FP8)**: **`44.0%` Official Kaggle Pass@1** / **`44.9%` Official Kaggle Pass@2** in `10.93M` total tokens (`~4.8h` on `4×L4`).
3. **`qwen_harness/` (`google3/.../kaggle/arc_agi_2/{pilot_1,pilot_2,notebooks}/`) — Havu Candidate B32 Port & Agentic Python Harnesses**:
   - Faithful port of Havu's **Candidate B32** harness (`havu_b32_harness.py`, `havu_auto_repair.py`, `havu_sandbox_zygote.py`, `prompts_and_selector.py`) supporting both DeepSeek (`DSML`) and Qwen (`ChatML`) tool calling.

---

## 📊 Technical Reports (`reports/`)

* **[`reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md`](reports/DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md)** — **START HERE: Complete Reference for Pruned DeepSeek-V4.1 (`Tool E128` 2-Bit) SFT & QAT**:
  - Full model architecture (`40` layers, `MLA+SWA+Compressor+Indexer`, `mHC` 20-iter Sinkhorn, `128` routed experts + `1` shared expert, `64 GiB` Compact Engram on layers 1 & 14).
  - Exact quantization formats: `FP8 E4M3` + `UE8M0` power-of-two block-32 scales (`1D` vs `2D` layouts, cross-shard Layer 9 scale fix) and 2-bit Humming `uint2` asymmetric Cholesky-GPTQ (`q, s, z` packing with `g128` for `w1/w3` and `g64` for `w2`).
  - What is frozen (`29.48B` params) vs. trained (`1.87B` params/rank), how QAT (`--enable_qat=true`, FP32 master weights, STE fake-quantization, direct trainable group scales `s`, QAD Forward-KL) resolves the 2-bit dead-zone effect, dataset curation (`dsv41_e128_short_clean`, `3,642` train / `227` val trajectories), `WIC-alnum = 10.0` loss weighting, hyperparameters, and full `v12` loss & evaluation tables.
* **[`reports/qat_vs_deepseek_sft_summary_report.md`](reports/qat_vs_deepseek_sft_summary_report.md)** & **[`reports/qat_sft_nemotron_vs_dsv41_report.md`](reports/qat_sft_nemotron_vs_dsv41_report.md)** — Theoretical comparison of NVIDIA Nemotron NVFP4 QAT/QAD vs. 2-bit MoE SFT.
* **[`reports/deepseek_pruning_and_sft_report.md`](reports/deepseek_pruning_and_sft_report.md)** — Deep dive on Havu's `E043` pruning (`384 -> 128` experts with `w=10` tool-block protection, `188.8 -> 64.0 GiB` Engram census) and `uint2` GPTQ (`percdamp = 1.0`).
* **[`reports/dsv41_e128_sft_v12_audit_report.md`](reports/dsv41_e128_sft_v12_audit_report.md)** — Harness fidelity & checkpoint audit report.
* **[`reports/havu_b32_vs_pilot2_harness_comparison.md`](reports/havu_b32_vs_pilot2_harness_comparison.md)** & **[`reports/PILOT_2_EXPERIMENTS.md`](reports/PILOT_2_EXPERIMENTS.md)** — Comparison of Havu Candidate B32 vs. `pilot_2` (`h1–h8`).
* **[`reports/qwen_sft_harness_fix_mini_report.md`](reports/qwen_sft_harness_fix_mini_report.md)**, **[`reports/qwen_sft_lora_report.md`](reports/qwen_sft_lora_report.md)**, & **[`reports/ARC_AGI2_SFT_DATA_CATALOG.md`](reports/ARC_AGI2_SFT_DATA_CATALOG.md)** — Qwen3.8-27B Full FSDP2 SFT & LoRA SFT reports and dataset catalog.

---

## 📁 Code Organization (GitHub Branch `arazd` vs. Google3 Paths)

| GitHub Folder (`arazd` branch) | Google3 Package Path | Contents |
| :--- | :--- | :--- |
| **[`deepseek_sft_qat/`](deepseek_sft_qat/)** | `//experimental/users/arazdai/kaggle/sft` | `dsv41_sft_model.py`, `train_dsv41_sft.py`, `eval_dsv41_b32_arc2.py`, `launch_dsv41_sft_b200_xm.py`, `launch_dsv41_b32_eval_xm.py`, `load_dsv41_2bit_example.py`, `inspect_and_download_chankhavu_e128.py`, `data_pipeline.py`, `audit_and_prepare_data.py`, `sft_test.py` |
| **[`qwen_sft/`](qwen_sft/)** | `//experimental/users/arazdai/kaggle/sft` | `train_full_sft.py`, `train_sft.py`, `fast_deltanet.py`, `export_and_quantize.py`, `launch_full_sft_b200_xm.py`, `launch_sft_b200_xm.py`, `evaluate_difficulty_argon.py`, `e031_audit_summary.json` |
| **[`qwen_harness/`](qwen_harness/)** | `//experimental/users/arazdai/kaggle/arc_agi_2` | `pilot_2/` (`havu_b32_harness.py`, `havu_auto_repair.py`, `havu_sandbox_zygote.py`, `prompts_and_selector.py`, `pilot2_eval.py`, `sandbox_verifier.py`), `pilot_1/` (`setB/`), `notebooks/`, `vllm_cuda_shim.py`, `vllm_fla_triton_ops.py` |
| **[`reports/`](reports/)** | `//experimental/users/arazdai/kaggle/arc_agi_2/reports` | `DEEPSEEK_SFT_QAT_COMPLETE_REPORT.md` + DeepSeek & Qwen technical reports |

---

## 🚀 1. How to Load a 2-Bit DeepSeek Checkpoint & Generate (`load_dsv41_2bit_example.py`)

Both Havu's base checkpoint (`/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit`) and every exported SFT/QAT milestone (`.../export_2bit/epoch_0p25`, `epoch_0p50`, `epoch_0p75`, `epoch_1p00`) share the exact same on-disk format:
- `model-0000{1,2}-of-00002.safetensors`: `FP8 E4M3` + `UE8M0` block-32 backbone (`330` FP8 tensors + `594` BF16/FP32 tensors).
- `gptq/layer_00..39.safetensors`: 2-bit Humming `uint2` routed experts (`e{0..127}.{w1,w2,w3}.{q,s,z}`, `640` local experts unpacked per GPU across `8` Expert-Parallel ranks).
- `model-engram-*.safetensors`: Compact Engram tables for layers `1` and `14` (`6` slices loaded per GPU from `--engram_model_dir`).

See [`load_dsv41_2bit_example.py`](../sft/load_dsv41_2bit_example.py) for the minimal standalone script:

```python
# Core loading & greedy generation snippet (runs across 8 GPUs via torchrun --nproc_per_node=8):
with gfile.Open(os.path.join(ckpt_dir, "config.json"), "r") as f:
  cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
cfg.routed_expert_lora_r = 0          # adapters already merged into 2-bit weights
cfg.enable_qat = False                # dequantized BF16 forward
cfg.load_compact_engram = True        # load layers 1 & 14 Engram tables

with torch.device(device):
  model = dsv41_sft_model.Dsv41SftModel(
      cfg=cfg, ep_rank=rank, ep_world_size=world_size,
      gradient_checkpointing=False, ep_replicated_input=True,
  )
stats = dsv41_sft_model.load_dequantized_dsv41_e128(
    model_dir=ckpt_dir,
    model=model,
    device=device,
    engram_model_dir="/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
)
assert stats["fp8_dequantized_tensors"] == 330
model.eval()
```

Run it via Blaze:
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents build \
  //experimental/users/arazdai/kaggle/sft:load_dsv41_2bit_example
```

---

## 🎯 2. How to Run ARC-AGI-2 Evaluation with Havu's Best Harness (`Candidate B32`)

[`eval_dsv41_b32_arc2.py`](../sft/eval_dsv41_b32_arc2.py) loads any 2-bit checkpoint onto `8×B200` (`EP=8`), runs a batched KV-cached decode loop (`24` concurrent streams per 8-GPU shard, `max_model_len = 163,840`), and drives [`havu_b32_harness.py`](pilot_2/havu_b32_harness.py) (`N=2` attempts per task, `80k` hard / `40k` gentle budget, multi-test scaling, AST/OSA ≤ 2 auto-repair, 4-path `train_check`, and `ranked_answers`).

### A. Launch a 4-Shard Candidate B32 Evaluation on XManager (Recommended — `~5.5h` wall time per milestone)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_b32_eval_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_e128_sft_v12_b32_eval_4shard \
  --sft_output_root=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/v12_fixed_oom_moe_only_full_sft/wu_1 \
  --eval_output_root=/cns/lu-d/home/arazdai/kaggle/sft/evals/v12_fixed_oom_moe_only_full_sft_b32_4shard \
  --milestones=0.25,0.50,0.75,1.00 \
  --num_shards=4 \
  --max_concurrent_streams=24 \
  --num_attempts=2
```

### B. Evaluate a Single Checkpoint or Base Model Directly
```bash
# Evaluate the unmodified base 2-bit model across 4 shards:
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_b32_eval_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_e128_base_b32_eval \
  --eval_base_model=true \
  --base_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --eval_output_root=/cns/lu-d/home/arazdai/kaggle/sft/evals/dsv41_e128_base_b32 \
  --num_shards=4
```

---

## 🔥 3. How to Run SFT & QAT on Pruned DeepSeek (`8×B200`)

### A. Launch MoE-Only SFT (`v12` Recipe — Shared Experts + Routed Expert LoRA `r=32`, Preemption Auto-Resume, Milestone 2-Bit Exports)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_sft_b200_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_e128_moe_only_full_sft_8xb200_v12 \
  --model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --data_dir=/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean \
  --output_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/v12_fixed_oom_moe_only_full_sft/wu_1 \
  --train_moe_only=true \
  --enable_qat=false \
  --load_compact_engram=true \
  --routed_expert_lora_r=32 \
  --routed_expert_lora_alpha=64 \
  --learning_rate=1e-5 \
  --num_epochs=1.0 \
  --grad_accum_steps=4 \
  --max_seq_len=36864 \
  --export_2bit_milestones=true \
  --milestone_fractions=0.25,0.50,0.75,1.00 \
  --auto_resume=true
```

### B. Launch Full Quantization-Aware Training (`QAT` + `STE` + Direct Trainable 2-Bit Group Scales + `QAD` Forward-KL)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_dsv41_sft_b200_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=dsv41_e128_moe_qat_ste_8xb200 \
  --model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
  --data_dir=/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean \
  --output_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/qat_ste_moe_only/wu_1 \
  --train_moe_only=true \
  --enable_qat=true \
  --qat_trainable_expert_scales=true \
  --qat_kl_weight=0.1 \
  --load_compact_engram=true \
  --routed_expert_lora_r=32 \
  --routed_expert_lora_alpha=64 \
  --learning_rate=1e-5 \
  --lora_learning_rate=5e-5 \
  --num_epochs=1.0 \
  --grad_accum_steps=4 \
  --max_seq_len=36864 \
  --export_2bit_milestones=true \
  --milestone_fractions=0.25,0.50,0.75,1.00 \
  --auto_resume=true
```

### C. Run Unit Tests (`sft_test` & `havu_b32_harness_test`)
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
  //experimental/users/arazdai/kaggle/sft:sft_test \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:havu_b32_harness_test \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_test
```
