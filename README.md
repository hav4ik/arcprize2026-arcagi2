# ARC-AGI-2: Agentic Python Synthesis + Qwen3.8-27B Full FSDP2 & LoRA SFT

How far can a single open-weights 27B model go on **ARC-AGI-2** (`120` official evaluation tasks, `167` test pairs) when combined with self-verifying Python synthesis and supervised fine-tuning (SFT) on 3-gate verified trajectories?

1. **Qwen3.8-27B Full-Parameter FSDP2 SFT (`26.9B`, Block-FP8 Weights) + Fixed `E031` Harness**:
   - **Full SFT Step 100 (`60k` Cap, `N=2`)**: Reached **`55.00%` Official Kaggle Pass@2** (`66.0/120` tasks; **`50.00%` Pass@1**) and **`51.50%` Pair-Level Pass@2** (`86/167` pairs; **`47.31%` Pass@1**) — a **+14.24 percentage point** jump after fixing the evaluation harness, outperforming both LoRA SFT (`47.08%`) and collaborator Vu's best pure-Qwen `E031/E032` baseline (`49.17%`).
   - **Full SFT Step 192 Final (`42k` Fast `<6h` Cap, `N=2`)**: Reached **`44.86%` Official Kaggle Pass@2** (`53.8/120` tasks; **`44.03%` Pass@1**) and **`44.91%` Pair-Level Pass@2** in **`10.93M` total tokens (`~4.8 hours` on 4×L4 GPUs)** — comfortably inside Kaggle's **6-hour (`13.6M` token) runtime limit**.
2. **LoRA SFT (`r=64, α=128`, `435M` Trainable Params, Block-FP8 Weights)**:
   - Reached **`47.08%` Official Kaggle Pass@2** (`42.08%` Pass@1) at the `60k` cap and **`43.19%` Official Kaggle Pass@2** (`41.67%` Pass@1) at the fast `42k` `<6h` cap (`11.21M` tokens).
3. **Pure Base Model Harness Progression (`pilot_2`, `N=4` 2-Wave Gated)**:
   - Scaled unmodified `Qwen3.8-27B-FP8` from **`19.17%` $\to$ `49.15%` Official Task Pass@2** (`h1` $\to$ `h8`).

---

## 📊 Reports (`reports/`)

All experiment reports and catalogs are organized in the **[`reports/`](reports/)** folder:

* **[`reports/dsv41_e128_sft_v12_audit_report.md`](reports/dsv41_e128_sft_v12_audit_report.md)** — **Pruned DeepSeek-V4.1 (`Tool E128` 2-bit) SFT v12 + Candidate-B32 eval audit (2026-10-29)**: harness fidelity vs. Havu's B32 (one gap: missing minimal effort header), model identity + Havu baselines (E043 72.8/82.5, E046 73.7/84.6), our base-eval numbers/xids/ETAs, step-by-step SFT pipeline and ranked bug list (blocker: the 2-bit export is byte-identical to the base — the LoRA is lost at re-quantization), transcript legitimacy checks.
* **[`reports/deepseek_pruning_and_sft_report.md`](reports/deepseek_pruning_and_sft_report.md)**, **[`reports/qat_vs_deepseek_sft_summary_report.md`](reports/qat_vs_deepseek_sft_summary_report.md)**, **[`reports/qat_sft_nemotron_vs_dsv41_report.md`](reports/qat_sft_nemotron_vs_dsv41_report.md)** — How Havu pruned/quantized DeepSeek-V4.1-Flash-1017 to `E128` 2-bit, and how to SFT it (de-quantize → train → re-quantize) vs. QAT/QAD.
* **[`reports/havu_b32_vs_pilot2_harness_comparison.md`](reports/havu_b32_vs_pilot2_harness_comparison.md)** — The 12 architectural differences between Havu's Candidate B32 and `pilot_2`, with the Qwen3.8-27B R2-A results (+13.6 pp Pass@2 from the harness alone) and the fidelity audit of our port.
* **[`reports/qwen_sft_harness_fix_mini_report.html`](reports/qwen_sft_harness_fix_mini_report.html)** & **[`reports/qwen_sft_harness_fix_mini_report.md`](reports/qwen_sft_harness_fix_mini_report.md)** — **Post-Harness Fix Mini-Report (`55.00%` Official Kaggle Pass@2 / `51.50%` Pair Pass@2)**: Compares Comparable Baselines, LoRA SFT, and Full SFT (all in Block-FP8 at `N=2`), includes median token counts, and explains the 5 evaluation harness fixes in plain English.
* **[`reports/qwen_sft_lora_report.html`](reports/qwen_sft_lora_report.html)** — Comprehensive interactive HTML report on Full FSDP2 SFT vs. LoRA SFT training architecture, Triton Gated DeltaNet kernels, validation loss curves, and baseline ablations.
* **[`reports/PILOT_2_EXPERIMENTS.md`](reports/PILOT_2_EXPERIMENTS.md)** — Controlled 8-stage (`h1`–`h8`) pure-model harness progression on `Qwen3.8-27B-FP8`.
* **[`reports/ARC_AGI2_SFT_DATA_CATALOG.md`](reports/ARC_AGI2_SFT_DATA_CATALOG.md)** — 26-dataset SFT catalog, 3-gate rollout verification stats, and Top-10K candidate selection breakdown.
* **[`reports/ARC_AGI2_EXPERIMENTS_FINDINGS.md`](reports/ARC_AGI2_EXPERIMENTS_FINDINGS.md)** & **[`reports/arc_agi_2_gemma4_eval_report.md`](reports/arc_agi_2_gemma4_eval_report.md)** — Initial `EXP-00`–`EXP-19` exploration and Gemma-4 (`31B Dense` & `26B MoE`) evaluation report.

---

## 1. Scoreboard: Comparable Baselines vs. LoRA SFT vs. Full SFT (`N=2`, Block-FP8 Weights)

All runs below use **Block-FP8 (`FP8 E4M3`, `128×128` block scales) model weights** and **2 attempts per puzzle (`N=2`)** across all **120 ARC-AGI-2 tasks (`167` test pairs)**:

| Model / Checkpoint (`FP8` Weights) | Token Cap | Official Kaggle Pass@1 (120 Tasks) | Official Kaggle Pass@2 (120 Tasks) | Pair-Level Pass@1 (167 Pairs) | Pair-Level Pass@2 (167 Pairs) | Median Tokens / Try *(Solved Median)* | Total Eval Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Comparable Baseline (`h1`, Base FP8)** | `48k` | `18.33%` (`22.0/120`) | **`19.17%`** (`23.0/120`) | `16.77%` (`28/167`) | `17.96%` (`30/167`) | `25,015` *(`28,649`)* | `8.82M` |
| **2. Collaborator Base (`E020 v4`, Base FP8)** | `48k` | `34.17%` (`41.0/120`) | **`37.08%`** (`44.5/120`) | — | — | `~31,400` | `10.50M` |
| **3. Collaborator Base (`E031/E032`, Base FP8)** | `60k` | `42.92%` (`51.5/120`) | **`49.17%`** (`59.0/120`) | — | — | `~38,900` | `13.00M` |
| **4. LoRA SFT Step 150 (`r=64`, FP8)** | `60k` | `37.92%` (`45.5/120`) | **`43.61%`** (`52.3/120`) | `37.72%` (`63/167`) | `43.11%` (`72/167`) | `59,842` *(`25,490`)* | `15.59M` |
| **5. LoRA SFT Step 191 Final (`r=64`, FP8)** | `60k` | `42.08%` (`50.5/120`) | **`47.08%`** (`56.5/120`) | `40.12%` (`67/167`) | `44.91%` (`75/167`) | `59,652` *(`21,301`)* | `14.88M` |
| **6. Full SFT Step 192 Final (`26.9B`, FP8)** | `60k` | `44.17%` (`53.0/120`) | **`50.00%`** (`60.0/120`) | `43.11%` (`72/167`) | `49.10%` (`82/167`) | `46,879` *(`21,376`)* | `13.58M` |
| **7. Full SFT Step 100 (`26.9B`, FP8) ★ BEST** | `60k` | **`50.00%`** (`60.0/120`) | **`55.00%`** (`66.0/120`) | **`47.31%`** (`79/167`) | **`51.50%`** (`86/167`) | `53,478` *(`22,818`)* | `14.41M` |
| **8. Full SFT Step 192 Final (Fast `<6h` Cap)** | `42k` | **`44.03%`** (`52.8/120`) | **`44.86%`** (`53.8/120`) | **`43.71%`** (`73/167`) | **`44.91%`** (`75/167`) | `41,744` *(`18,002`)* | **`10.93M`** (`~4.8h`) |
| **9. LoRA SFT Step 191 Final (Fast `<6h` Cap)** | `42k` | `41.67%` (`50.0/120`) | **`43.19%`** (`51.8/120`) | `40.12%` (`67/167`) | `42.51%` (`71/167`) | `41,886` *(`21,689`)* | **`11.21M`** (`~4.9h`) |

---

## 2. Repository Structure

```text
.
├── README.md                          # Overview, scoreboard, repository layout & CLI commands
├── BUILD                              # Bazel/Blaze build targets for GPU/TPU Colab kernels & shims
├── vllm_cuda_shim.py                  # Custom CUDA/FlashAttn/Triton shim for Qwen3.8-27B on H100/B200
├── vllm_fla_triton_ops.py             # Fused Triton linear-attention ops for vLLM inference
├── vllm_h100_gpu_test.py              # Smoke test for Qwen3.8-27B vLLM GPU inference
│
├── reports/                           # All Markdown & Interactive HTML Experiment Reports
│   ├── qwen_sft_harness_fix_mini_report.html   # Mini-report: 55.0% Pass@2 SFT results & 5 harness fixes
│   ├── qwen_sft_harness_fix_mini_report.md     # Markdown version of the post-fix mini-report
│   ├── qwen_sft_lora_report.html               # Full interactive SFT & baseline benchmark report
│   ├── PILOT_2_EXPERIMENTS.md                  # Controlled h1-h8 harness engineering report
│   ├── ARC_AGI2_SFT_DATA_CATALOG.md            # SFT dataset catalog & 3-gate verification audit
│   ├── ARC_AGI2_EXPERIMENTS_FINDINGS.md        # Pilot 1 (EXP-00 to EXP-19) findings
│   ├── arc_agi2_experiments_report_mistake_analysis.html
│   ├── arc_agi_2_gemma4_eval_report.md         # Gemma-4 31B Dense vs 26B MoE benchmark
│   └── arc_agi2_gemma4_experiment_report.html
│
├── sft/                               # Qwen3.8-27B Full FSDP2 SFT & LoRA SFT Pipeline + Pruned DeepSeek-V4.1 (E128 2-bit) SFT
│   ├── BUILD                          # Blaze targets for data prep, training, FP8 export & tests
│   ├── README.md                      # Detailed SFT architecture & training documentation
│   ├── data_pipeline.py               # 3-gate filtering, D8 de-contamination, 90/10 split, ChatML + DeepSeek DSML rendering & token weights
│   ├── audit_and_prepare_data.py      # CLI to build train/val Parquet & JSONL / npz datasets
│   ├── fast_deltanet.py               # Custom @triton.jit chunk-recurrent Gated DeltaNet kernels
│   ├── train_full_sft.py              # 8×B200 Native PyTorch FSDP2 (26.9B) trainer + parallel FP8 export
│   ├── train_sft.py                   # 1×B200 LoRA (r=64, α=128) trainer with chunked cross-entropy
│   ├── export_and_quantize.py         # FP32 merge + 128×128 block-wise FP8 (float8_e4m3fn) quantizer
│   ├── launch_full_sft_b200_xm.py     # XManager launcher for 8×B200 Full FSDP2 SFT
│   ├── launch_sft_b200_xm.py          # XManager launcher for B200 LoRA SFT
│   ├── dsv41_sft_model.py             # Pruned DeepSeek-V4.1 E128: 2-bit/FP8 de-quant, differentiable forward (MLA/DSA, mHC, Engram, EP MoE, LoRA, QAT/STE), KV-cached decode, 2-bit Humming exporter
│   ├── train_dsv41_sft.py             # 8×B200 expert-parallel SFT trainer for E128 (MoE-only / QAT / QAD options, milestone 2-bit exports)
│   ├── launch_dsv41_sft_b200_xm.py    # XManager launcher for the E128 SFT
│   ├── eval_dsv41_b32_arc2.py         # 8×B200 PyTorch step engine that runs Havu's Candidate B32 harness on exported E128 checkpoints
│   ├── launch_dsv41_b32_eval_xm.py    # XManager launcher for milestone B32 evals (polls for EXPORT_COMPLETED.json)
│   ├── gen_dsv41_before_after.py      # Before/after SFT sanity check (teacher-forced NLL + greedy generations)
│   ├── inspect_and_download_chankhavu_e128.py  # Streams chankhavu's Kaggle E128 2-bit model to CNS
│   ├── repair_dsv41_export_shared_scales.py    # One-off repair of exports that lost cross-shard FP8 scales
│   ├── e031_audit_summary.json        # Verified dataset audit summary
│   └── sft_test.py                    # Unit tests for loss masking, split isolation, FP8/2-bit quantization, E128 model parity
│
├── pilot_2/                           # Main Agentic Python Evaluation Harness (TPU & GPU vLLM) + Havu Candidate B32 port
│   ├── BUILD                          # Blaze targets for pilot2_eval, launch_pilot2_xm & tests
│   ├── prompts_and_selector.py        # System/user prompts (tier=xhigh, XML tools) & candidate selector
│   ├── sandbox_verifier.py            # Stateful IPython sandbox, verify_submit & session salvage
│   ├── arc_dsl_and_summary.py         # Grid connected-components/symmetry analyzer & DSL helpers
│   ├── pilot2_eval.py                 # Async vLLM multi-turn evaluation loop (2-stage nudge, compaction)
│   ├── havu_b32_harness.py            # Port of Havu's Candidate B32 (multi-test, token-exact, auto-repair, 4-path train_check, ranked_answers) for Qwen (ChatML) and DeepSeek (DSML)
│   ├── havu_auto_repair.py            # Havu's zero-token identifier auto-repair (OSA ≤ 2, AST side-effect scan)
│   ├── havu_sandbox_zygote.py         # Pre-TPU/GPU fork server providing isolated sandbox processes on Borg
│   ├── auto_repair_names.json         # Module/attribute name index for auto-repair
│   ├── havu_b32_harness_test.py       # 35 vLLM-free tests of the B32 port
│   ├── launch_pilot2_xm.py            # XManager multi-shard TPU evaluation launcher
│   └── pilot2_test.py                 # Unit tests for prompts, XML parser, sandbox & selector
│
├── pilot_1/                           # Pilot 1 Harnesses & DeepSeek-V4.1-Flash Rollout Generators
│   └── setB/
│       ├── eval_qwen_vllm_h100.py     # H100 vLLM evaluation script
│       ├── launch_qwen_arc_h100_xm.py # H100 XManager launcher
│       ├── run_arc_eval.py            # Multi-model API & vLLM runner
│       └── tpu_vllm/                  # TPU vLLM & DeepSeek-V4.1-Flash-1017 Top-10K rollout generator
│
└── notebooks/                         # Kaggle 4×L4 Submission Notebooks & Scheduler Patches
    ├── arc2-qwen38-e032-sft-minimal.ipynb
    ├── arc2-qwen38-e032-sft-ensemble4.ipynb
    ├── arc2_qwen38_tools_v3_ensemble.ipynb
    └── cell21_patch.py
```

---

## 3. How to Run Commands (Unit Tests, Data Prep, SFT Training, FP8 Export & Evaluation)

All commands below are run from the `google3/` workspace root using `/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents` and `/google/bin/releases/xmanager/cli/xmanager.par`.

### A. Run Unit Tests (`pilot_2` Harness & `sft` Pipeline)
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_test \
  //experimental/users/arazdai/kaggle/sft:sft_test
```

### B. Audit & Prepare SFT Training Data (`1,535` Train / `64` Val Verified Trajectories)
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run \
  //experimental/users/arazdai/kaggle/sft:audit_and_prepare_data -- \
  --input_parquet=/cns/lu-d/home/arazdai/kaggle/arc_agi_2/sft_data/e031_deepseek_rollouts.parquet \
  --output_dir=/cns/lu-d/home/arazdai/kaggle/sft/prepared_e031
```

### C. Launch 8×B200 Full-Parameter FSDP2 SFT (`26.9B` Params, `75.5 min` for 192 Steps + Auto Block-FP8 Export)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_full_sft_b200_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=qwen38_27b_arc_agi2_e031_full_sft_8xb200
```

### D. Launch B200 LoRA SFT (`r=64, α=128`, `435M` Params)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/sft:launch_sft_b200_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --exp_name=qwen38_27b_arc_agi2_e031_lora_r64
```

### E. Standalone Merge & Export Any Checkpoint to Block-FP8 (`float8_e4m3fn`, `128×128` Blocks)
```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run \
  //experimental/users/arazdai/kaggle/sft:export_and_quantize -- \
  --base_model_path=/cns/lu-d/home/arazdai/models/Qwen3.8-27B \
  --adapter_or_ckpt_path=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/step-00100 \
  --output_fp8_dir=/cns/lu-d/home/arazdai/kaggle/sft/exported_fp8/step-00100
```

### F. Launch Multi-Shard ARC-AGI-2 Evaluation (`e031_deepseek_teaching` & `e031_deepseek_teaching_fast`)
```bash
/google/bin/releases/xmanager/cli/xmanager.par launch \
  //experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:launch_pilot2_xm -- \
  --noxm_monitor_on_launch \
  --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
  --xm_skip_launch_confirmation \
  --experiment_title=arc_agi2_full_sft_step100_xhigh_e031 \
  --exp_ids=e031_deepseek_teaching,e031_deepseek_teaching_fast \
  --num_shards=14 \
  --model_path=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_full_sft_v2/xid_304470493_qwen38_27b_arc_agi2_e031_full_sft_8xb200_v3/wu_1/fp8_checkpoints/step-00100 \
  --cns_output_root=/cns/lu-d/home/arazdai/kaggle/arc_agi_2/eval_runs/full_sft_step100
```
