# ARC-AGI-2: Agentic Python Synthesis + Qwen3.8-27B Full FSDP2 & LoRA SFT

How far can a single open-weights 27B model go on **ARC-AGI-2** (`120` public evaluation tasks, `167` test pairs) when combined with self-verifying Python synthesis and supervised fine-tuning (SFT) on 3-gate verified trajectories?

1. **Pure Harness Engineering (`pilot_2`, `N=4` 2-Wave Gated)**: Scaled **`Qwen3.8-27B-FP8`** from **`19.17%` $\to$ `49.15%` Official Task Pass@2** (`57.49%` Pair Pass@2 across our multi-view ensemble).
2. **Qwen3.8-27B LoRA SFT & Full 27B FSDP2 SFT ([`sft/`](sft/), `N=2` Comparable `lean2sub` Harness)**:
   - **LoRA SFT (`r=64`, Step 150)**: Boosted `N=2` Official Task Pass@2 from **`38.94%` $\to$ `46.94%` (`+8.00 pp`)** and Pass@1 from **`36.39%` $\to$ `42.29%` (`+5.90 pp`)**, outperforming even the `N=4` `h6` Hybrid Ensemble (`36.50%`) with half the attempts.
   - **Full-Parameter 26.9B FSDP2 SFT (`Step 100` at 40k Cap)**: Achieved **`40.76%` Official Task Pass@2** (`+1.82 pp` over the `38.94%` baseline) while generating **`10.35M` total tokens (`~4.6h` on `4×L4`)**—comfortably inside Kaggle's **6-hour (`13.6M` token) budget** and reaching the lowest held-out validation loss of any run (**`0.7549`** at Step 192 in just **75.5 minutes** on `8×B200` via custom Triton Gated DeltaNet kernels).

📄 **Full Interactive HTML Benchmark Report**: **[`sft/qwen_sft_lora_report.html`](sft/qwen_sft_lora_report.html)** (also at root: **[`qwen_sft_lora_report.html`](qwen_sft_lora_report.html)**)

---

## 1. Qwen3.8-27B Full SFT vs. LoRA SFT vs. Comparable Baseline (`sft/`)

All runs below are evaluated on all **120 official ARC-AGI-2 tasks (`167` test pairs)** using the **Comparable SFT-Aligned (`lean2sub`) Harness** ($N=2$ attempts per pair, `334` total attempts, native XML `<function=run_python>` + `<function=submit_answer>`):

| Model / Run | Train Loss | Held-Out Val Loss (PPL) | **Official Kaggle Pass@1** | **Official Kaggle Pass@2** | Median Tokens / Try *(Solved)* | Total Tokens (`167×2` tries) | Fits Kaggle 6h (`4×L4` $\le 13.6\text{M}$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Qwen3.8-27B Baseline** *(Comparable `lean2sub`)* | — | `0.8719` (`2.392`) | `36.39%` (`42.9/118`) | **`38.94%`** (`46.0/118`) | `25,008` *(`16,133`)* | `11.16M` | **YES (`~4.9h`, `82%` cap)** |
| **Best LoRA SFT (`Step 150`, `r=64`)** *(80k Cap)* | `0.5800` | `0.7859` (`2.194`) | **`42.29%`** (`50.3/119`) | **`46.94%`** (`55.9/119`) | `56,423` *(`26,349`)* | `16.84M` | **NO (`~7.4h`, `124%` cap)** |
| **Best Full SFT (`Step 100`, `26.9B`)** *(40k Cap)* | `0.6254` | `0.7689` (`2.157`) *(Step 192: `0.7549`)* | `37.17%` (`42.0/113`) | **`40.76%`** (`46.1/113`) | `39,713` *(`17,578`)* | **`10.35M`** | **YES (`~4.6h`, `76%` cap)** |

### SFT Codebase & Scripts ([`sft/`](sft/))
* **[`sft/train_full_sft.py`](sft/train_full_sft.py)** — **8×B200 Native PyTorch `FSDP2` (`fully_shard`) Full-Parameter (`26.9B`) Trainer**: `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, DTensor `foreach=True` AdamW (`lr=2e-5`), global token-weighted loss reduction, distributed checkpoints, and parallel 8-GPU Block-128 FP8 export (`75.5 min` for 192 steps).
* **[`sft/fast_deltanet.py`](sft/fast_deltanet.py)** — **Custom `@triton.jit` Fused Chunk-Recurrent Gated DeltaNet Kernels (`BT=64`)**: Replaces HuggingFace's sequential Python loop across Qwen3.8's 48 linear-attention layers with a vectorized cuBLAS triangular solver and fused forward/backward Triton state recurrence (`5–8×` speedup).
* **[`sft/train_sft.py`](sft/train_sft.py)** — **1×B200 LoRA SFT (`r=64, alpha=128`) Trainer**: Targets all 10 attention, Gated DeltaNet, and MLP projections (`435.2M` trainable params) with FP32 master weights and chunked cross-entropy (`chunk_size=2048`).
* **[`sft/audit_and_prepare_data.py`](sft/audit_and_prepare_data.py)** & **[`sft/data_pipeline.py`](sft/data_pipeline.py)** — 3-gate rollout verification (`correct & train_pass & submit`), $D_8$+color isomorphism eval de-contamination, parent-grouped 90/10 split (`0` parent overlap), and token-exact assistant-only loss mask verification (`verify_token_loss_mask`).
* **[`sft/export_and_quantize.py`](sft/export_and_quantize.py)** & **[`sft/sft_test.py`](sft/sft_test.py)** — Official $128\times 128$ block-wise FP8 (`float8_e4m3fn`) merger/quantizer and end-to-end unit test suite.
* **[`sft/qwen_sft_lora_report.html`](sft/qwen_sft_lora_report.html)** — Complete interactive HTML report with all tables, SVG plots, training architecture details, and comparison to Havu's Tinker API experiments (`E028` / `E032`).

---

## 2. Pure Model Harness Progression (`pilot_2`: `Qwen3.8-27B-FP8` on All 120 Tasks)

Each row in `pilot_2` locks the exact same model and sampling parameters (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `131k` context) and changes **one controlled axis at a time**:

| Stage | What We Changed | Attempts | Turns | **Task Pass@1** | **Task Pass@2 (Kaggle)** | Fits `4×L4` (`6h` = `13.6M`)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`h1` Baseline** | Basic Python REPL (`lean2prog` raw grids, JSON tool calls) | `2` | `20` | `18.3%` | **`19.2%`** (`23.0/120`) | Yes (`8.00M` / `~3.5h`) |
| **`h2` + Attempts** | Double candidate trajectories (`N=2 -> N=4`) + majority vote | `4` | `20` | `24.3%` | **`26.4%`** (`31.7/120`) | No (`15.96M` / `~7.0h`) |
| **`h3` + 40 Turns** | **40-turn loop + auto `check(solve)` cell-coordinate diffs** | `4` | `40` | `32.2%` | **`33.3%`** (`40.0/120`) | No (`21.48M` / `~9.5h`) |
| **`h4` Homogeneous DSL** | Pre-computed CC/symmetry summary + `arc_helpers` on *all 4* | `4` | `40` | `26.0%` | **`28.8%`** *(hurt diversity)* | No (`22.66M` / `~10.0h`) |
| **`h5` Homogeneous Roles**| Role-based falsification (`lean2progroles`, `64k` cap) on *all 4* | `4` | `40` | `28.3%` | **`31.9%`** (`38.3/120`) | No (`28.05M` / `~12.4h`) |
| **`h6` Hybrid Ensemble** | **2× Fast Raw + 2× Deep Role/DSL + $D_8$/Color Equivariance** | `4` | `40` | `35.3%` | **`36.5%`** (`43.8/120`) | No (`25.25M` / `~11.1h`) |
| **`h7` Native XML** | **Native XML `<function=run_python>` + 2-Stage Graceful Nudge** | `4` | `40` | **`43.8%`** | **`47.4%`** (`53.5/113`) | No (`36.93M` / `~16.3h`) |
| **`h8` 2-Wave Repair** | **Wave-2 Evolutionary Near-Miss Code Repair + `<think>` Compaction** | `4` | `40` | `41.9%` | **`49.2%`** (`57.5/117`) | No (`31.49M` / `~13.9h`) |
| **`EXP-19` Multi-View** | Offline 12-attempt Text + Multimodal RGB Pool + Smart Rating | `12` | `16–40`| **`49.7%`** *(pair)* | **`57.5%`** *(96/167 pairs)* | Offline (`12×48k`) |

---

## 3. Gemma-4 (`31B Dense` & `26B MoE`) vs. Qwen3.8-27B

We also benchmarked **Gemma-4** (`31B Dense` and `26B MoE` with `4B` active parameters) across all 120 tasks:

| Run | Model | Setup | **Pair Pass@1 / Pass@2** | **Task Pass@1 / Pass@2** | Key Finding |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Run 1** | `Gemma-4 26B MoE` (`4B` active) | High Thinking, No Tools (`N=2`) | `1.2% / 1.8%` | `1.1% / 1.9%` | `4B` active params cannot track 2D grid coordinates in working memory. |
| **Run 3** | `Gemma-4 31B Dense` | High Thinking, No Tools (`N=2`) | `6.0% / 8.4%` | `6.5% / 9.3%` | Dense 31B solves `5×` more tasks, but stops thinking around `~11k` tokens. |
| **Run 4** | `Gemma-4 31B Dense` | High Thinking + Continuation (`196k`) | `7.8% / 10.8%` | `7.5% / 11.7%` | Extra token budget alone barely moves accuracy without code execution. |
| **Run 5** | `Gemma-4 31B Dense` | **Python REPL + Train Check** (`N=2`) | `16.2% / 21.6%` | `16.8% / 22.1%` | **2× jump**: testing `solve(grid)` on `train` pairs doubles accuracy. |
| **Run 6** | `Gemma-4 31B Dense` | Python REPL + Homogeneous Prompt Tricks | `13.2% / 16.2%` | `12.1% / 15.8%` | Over-prompting increases `train` overfitting (`58% -> 45%` test precision). |
| **Run 8** | `Gemma-4 31B Dense` | **RGB Image + Python REPL + `scipy` (`N=3`)** | **`25.3% / 27.3%`** | **`26.3% / 28.7%`** | **Best Gemma-4 setup**: 2D RGB grid images unlock `+6.6 pp` over text REPL. |

---

## 4. DeepSeek-V4.1-Flash (`DeepSeek-V4.1-Flash-1017` Full-Precision BF16$\to$FP32) on All 120 Tasks

We evaluated **`DeepSeek-V4.1-Flash-1017`** (`40`-layer Causal Encoder-Decoder + `196.6B` Engram conditional memory at `L1`/`L14` + `3`-layer DSpark `1+5` MTP drafter, full-precision `BF16 -> FP32` MXU accumulation, `Reasoning Effort: 100`) across all **`120` ARC-AGI-2 public evaluation tasks (`167` test pairs)** in two controlled stages ([`deepseek_v41_arc2_eval.py`](pilot_1/setB/tpu_vllm/deepseek_v41_arc2_eval.py)):

1. **`v1` SFT-Trace Extraction Harness**: Replicated the 5-turn SFT trace-extraction harness, reaching **`8.98%` Pair Pass@1 & Pass@2** (`15 / 167`) and **`9.17%` Task Pass@1 & Pass@2** (`11.00 / 120`).
   * **Root-Cause Diagnosis (`v1` vs. `Qwen3.8` `pilot_2`)**: Whenever `DeepSeek-V4.1-Flash` verified a candidate program on `train` (`train_passed=True`), held-out test accuracy was **`92.9%` (`13 / 14` pairs)**—higher than `Qwen3.8-27B` (`~68%`). However, **`77.2%` (`258 / 334`) of attempts never defined `def solve(grid):`** due to: (a) Turn 2+ Qwen ChatML `<|im_start|>`/`<|im_end|>` token pollution instead of native `<｜end▁of▁sentence｜><｜User｜><tool_result>`, (b) missing stop strings on `</tool_call>`/`</｜DSML｜ calls>` causing `30.8%` of attempts to hallucinate fake turns after tool calls, (c) `74.9%` of attempts hitting the `16,384` context window with uncompacted prior `<think>` blocks and spaced grids, and (d) `max_turns=5` with `0` repair rounds and `turn_think_cap=8000`.
2. **`v3` Native DS4.1 Agentic Harness (`ds41_fp_native_v3`)**: Fixed all multi-turn harness bugs while keeping **100% identical model weights, full precision (`BF16 -> FP32`), and `Reasoning Effort: 100`**, jumping accuracy **2.20×–2.60×** to **`19.76%` Pair Pass@1 / `23.35%` Pair Pass@2** (`33` / `39` of `167` pairs) and **`20.42%` Task Pass@1 / `23.75%` Task Pass@2** (`24.50` / `28.50` of `120` tasks).

---

## 5. Repository Map

* **[`sft/`](sft/)** — **Qwen3.8-27B Full FSDP2 SFT & LoRA SFT Codebase** (`train_full_sft.py`, `train_sft.py`, `fast_deltanet.py`, `data_pipeline.py`, `audit_and_prepare_data.py`, `export_and_quantize.py`, `sft_test.py`) + Interactive HTML Report ([`sft/qwen_sft_lora_report.html`](sft/qwen_sft_lora_report.html) & [`sft/README.md`](sft/README.md)).
* **[`pilot_2/`](pilot_2/)** — Clean, unit-tested TPU/GPU vLLM harness (`pilot2_eval.py`, `prompts_and_selector.py`, `sandbox_verifier.py`, `arc_dsl_and_summary.py`) and full `h1`–`h8` report ([`PILOT_2_EXPERIMENTS.md`](pilot_2/PILOT_2_EXPERIMENTS.md)).
* **[`pilot_1/`](pilot_1/)** — Initial `EXP-00`–`EXP-19` exploration ([`ARC_AGI2_EXPERIMENTS_FINDINGS.md`](pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md)), Gemma-4 API benchmark ([`arc_agi_2_gemma4_eval_report.md`](pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md)), and `DeepSeek-V4.1-Flash-1017` trace generator.
* **[`ARC_AGI2_SFT_DATA_CATALOG.md`](ARC_AGI2_SFT_DATA_CATALOG.md)** — 26-dataset SFT catalog, 3-level verification stats, and Top-10K candidate selection breakdown.
* **[`arc2_qwen38_tools_v3_ensemble.ipynb`](arc2_qwen38_tools_v3_ensemble.ipynb)** · **[`arc2-qwen38-e032-sft-ensemble4.ipynb`](arc2-qwen38-e032-sft-ensemble4.ipynb)** · **[`cell21_patch.py`](cell21_patch.py)** — Kaggle `4×L4` notebooks and 2-Wave Gated scheduler.
