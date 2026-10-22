# ARC-AGI-2 Experiments, Agentic Python-REPL Harness & SFT Catalog (`arc_agi_2`)

This directory contains our complete experimental suite, evaluation harnesses, Kaggle submission notebooks, and SFT data pipeline for **ARC-AGI-2** (`120` public evaluation tasks, `167` test pairs).

---

## 1. Repository Layout & Quick Links

| Path | Description |
| :--- | :--- |
| [`README.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/README.md) | **Master overview** (this document): experimental setup, models, hypotheses tested, Qwen3.8 & Gemma-4 results tables, and key lessons on what worked. |
| [`pilot_2/PILOT_2_EXPERIMENTS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/PILOT_2_EXPERIMENTS.md) | **Pilot 2 (`h1`–`h8`) Controlled Progression**: Strictly locked baseline (`FROZEN_BASELINE_CONFIG`) isolating single-factor gains from `19.17%` $\to$ **`49.15%` Official Task Pass@2**. |
| [`pilot_2/pilot2_eval.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/pilot2_eval.py) · [`prompts_and_selector.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/prompts_and_selector.py) · [`sandbox_verifier.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/sandbox_verifier.py) | Clean, unit-tested ([`pilot2_test.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/pilot2_test.py)) TPU vLLM evaluation harness, subprocess sandbox with cumulative helper replay, $D_8$/color equivariance audit, and candidate selector. |
| [`pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md) | **Pilot 1 (`EXP-00` to `EXP-19`) & Hard-Subset Ablations**: Initial full-set exploration, `dev_60` / `hard_50` / `hard_25` pilots, 4-attempt agreement stats, and mistake taxonomy. |
| [`pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md) | **Gemma-4 (`31B Dense` & `26B MoE`) vs. Qwen3.8-27B Report**: Head-to-head comparison across tool use, multimodal RGB grids, and thinking budgets. |
| [`ARC_AGI2_SFT_DATA_CATALOG.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/ARC_AGI2_SFT_DATA_CATALOG.md) | **SFT Data & 3-Level Verification Pipeline**: 26 standardized puzzle datasets (`200k+` tasks), 1K empirical screening, Top-10K frontier selection, and `DeepSeek-V4.1-Flash-1017` / `Qwen3.8-27B` 3-level verified trajectories. |
| [`arc2_qwen38_tools_v3_ensemble.ipynb`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/arc2_qwen38_tools_v3_ensemble.ipynb) · [`arc2-qwen38-e032-sft-ensemble4.ipynb`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/arc2-qwen38-e032-sft-ensemble4.ipynb) · [`cell21_patch.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/cell21_patch.py) | Kaggle `4×L4` (`96 GB` VRAM, `12h`) submission notebooks and 2-Wave Gated Early-Stopping scheduler (`ARC_ATTEMPT2="gated"`). |

---

## 2. Experimental Setup, Metrics & Hardware Constraints

### 2.1 Benchmark & Scoring Metrics
* **Dataset**: Official **ARC-AGI-2 Public Evaluation Set** (`/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation`):
  * **`120` tasks** containing **`167` test input-output pairs** (`75` tasks with 1 test pair, `43` tasks with 2 test pairs, `2` tasks with 3 test pairs: $75\times 1 + 43\times 2 + 2\times 3 = 167$).
  * Each test pair allows **2 submission attempts** (`attempt_1` and `attempt_2`).
* **Two Scoring Conventions Reported**:
  1. **Official ARC Prize / Kaggle Task-Weighted `Pass@1` & `Pass@2`** *(used in `pilot_2` and official leaderboards)*: Each of the `120` tasks has equal weight `1.0` (a task with $k$ test pairs awards $1/k$ per solved test pair).
  2. **Pair-Weighted `Pass@1` & `Pass@2`** *(used in `pilot_1` pair-level diagnostics)*: Fraction of the `167` total test pairs solved (`count / 167`).

### 2.2 Agentic Python-REPL Harness (`FROZEN_BASELINE_CONFIG`)
Across our controlled experiments, the model interacts with an isolated subprocess Python sandbox (`sandbox_verifier.py`) over up to $T \in \{16, 20, 40\}$ turns per attempt:
* **vLLM Serving**: `tensor_parallel_size=4`, `max_model_len=131072`, `gpu_memory_utilization=0.90`, `enable_prefix_caching=True` (append-only ChatML so KV cache hits 100% of prior turns).
* **Sampling**: `temperature=1.0`, `top_p=0.95`, `top_k=20`.
* **Verification & First-Verified Lock (`lock_first_verified_pred=True`)**: Whenever the model defines `def solve(grid):` (or calls `submit_program`), the harness executes `solve` on all demonstration `train` pairs in an isolated subprocess (stripping global `train`/`test_input` variables so `solve` cannot cheat, while replaying helper functions from prior turns). The **first** program that passes 100% of training pairs (`train_pass=True`) is locked as `first_tp_pred` so subsequent audit turns cannot overwrite it with unverified guesses.
* **Candidate Ranking (`select_top_two_predictions`)**:
  * Candidates are grouped into **Tier 0** (`train_pass=True`), **Tier 1** (`partial train pass`), and **Tier 2** (`unverified fallback`).
  * Ranked lexicographically by: `(verified_votes DESC, primary_votes DESC, total_votes DESC, peer_sim_support DESC, equivariance_bonus DESC, -min_gen_tokens [Occam's Razor] DESC, -min_turn DESC, -min_attempt_idx DESC)`.

### 2.3 Kaggle `4×L4` (`96 GB` VRAM, `12h` Wall-Clock) Budget
* On a single Kaggle `4×L4` node (`TP=4`, FP8 weights `~28.5 GB`, `~60 GB` KV cache), batched vLLM throughput with prefix caching is **`~310–360 output tok/s`** (**`~12.7M–14.8M` total tokens** in `12` hours).
* **Raw vs. 2-Wave Gated Execution (`ARC_ATTEMPT2="gated"`)**:
  * **Raw (`N=4` unconditional)**: Runs all 4 attempts on all 167 pairs (`~19.8M–24.2M` tokens $\to$ `15h–22h`, exceeding Kaggle's `12h` limit).
  * **Gated 2-Wave Scheduling**: Runs **Wave 1 (`N=2`)** first. If both Wave 1 attempts pass all `train` pairs and agree on the exact same test grid (`verified_votes == 2`, where empirical test accuracy is **`95%–100%`**), the harness **stops immediately at `N=2`**, allocating **Wave 2 (`Attempts 2 & 3`)** exclusively to ambiguous (`1-vs-1` tie) or unsolved (`0/2` verified) pairs. This cuts token usage by **`~45–48%`** (`~12.2M–12.7M` tokens, **`~9.5h–11.6h` on `4×L4`**) while preserving `>99%` of full 4-attempt accuracy.

---

## 3. Models Evaluated

| Model | Architecture & Precision | Serving Infrastructure | Primary Role in Experiments | Best Full-Set Score |
| :--- | :--- | :--- | :--- | :---: |
| **`Qwen3.8-27B-FP8`** | 27B Dense, Fine-Grained FP8 (`27.4 GB`) | vLLM on TPU v6e (`viperfish=2x2x1`) / TPU v7x (`ghostfish=2x2x1`) & `4×L4` / `H100` | Primary reasoning + Python program synthesis model across `pilot_1` & `pilot_2` | **`49.15%` Task Pass@2** (`h8` single 4-att run) / **`57.49%` Pair Pass@2** (`EXP-19` 12-att pool) |
| **`Qwen3.8-27B-e032-SFT`** | 27B Dense + Rank-32 LoRA SFT (Block-FP8 quantized) | Kaggle `4×L4` (`arc2-qwen38-e032-sft-*.ipynb`) | Fine-tuned on verified ARC Python-REPL trajectories with E027 `<think>` compaction | Kaggle `4×L4` submission (`~9.5h` gated) |
| **`Gemma-4 31B Dense`** (`gemma-4-31b-it`) | 31B Dense (`BF16`) | AI Studio / OnePlatform API (14 parallel worker streams) | Model family comparison across thinking budgets, Python REPL, and RGB multimodal grids | **`28.7%` Task Pass@2** (`27.3%` Pair Pass@2, Run 8 Multimodal + REPL) |
| **`Gemma-4 26B MoE`** (`gemma-4-26b-a4b-it`) | 26B Total / **`4B` Active** MoE | AI Studio / OnePlatform API | Active-parameter scaling baseline (`4B` active vs `31B` dense) | `1.9%` Task Pass@2 (`1.8%` Pair Pass@2) |
| **`DeepSeek-V4.1-Flash-1017`** | MoE + 196.6B Engram Memory (`BF16` dequant + `FP32` MXU) | 16× TPU v7x slices (`ds41_full_precision` Pallas megakernel) | High-precision teacher for generating 3-level verified SFT trajectories on Top-10K candidates | Generated **`1,122`** 3-level verified SFT trajectories |

---

## 4. Summary of Hypotheses Tested

### 4.1 Pilot 2: Controlled Single-Factor Progression (`h1` $\to$ `h8` on All 120 Tasks)
Every experiment in [`pilot_2`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/PILOT_2_EXPERIMENTS.md) locks `FROZEN_BASELINE_CONFIG` and changes **one controlled axis at a time**:

| ID | Controlled Hypothesis Tested | Verdict | Impact on Official Task `Pass@1` / `Pass@2` | Why It Happened |
| :--- | :--- | :---: | :--- | :--- |
| **`h1`** | **Baseline (`N=2`, `20` turns, `48k/8k` JSON tools, raw grids)** | **Baseline** | `18.33%` / **`19.17%`** | High rate of JSON string-escaping syntax errors and mid-sentence `<think>` cutoffs at `8k` tokens. |
| **`h2`** | **`+` More Attempts (`N=2 -> N=4`, `20` turns)** | **YES (`+7.22 pp`)** | `24.31%` / **`26.39%`** | Doubling candidate trajectories + behavioural majority voting filters out single-attempt noise. |
| **`h3`** | **`+` Deeper Repair Loop (`20 -> 40` turns + `auto_check` cell diffs)** | **YES (`+6.94 pp`)** | `32.22%` / **`33.33%`** | Automatically running `check(solve)` and returning `(r, c): got X, expected Y` diffs lets Qwen fix off-by-one bugs on turns 21–40. |
| **`h4`** | **`+` Homogeneous `task_summary` + `arc_helpers` on all 4 attempts** | **NO (`-4.58 pp` vs `h3`)** | `25.97%` / **`28.75%`** | Injecting the exact same pre-computed CC/symmetry summary into all 4 attempts **collapsed trajectory diversity**, causing all 4 attempts to pursue the same flawed framing. |
| **`h5`** | **`+` Homogeneous Role-Based Falsification (`lean2progroles`, `64k/12k`)** | **PARTIAL (`-1.39 pp` vs `h3`)** | `28.33%` / **`31.94%`** | Solved harder compositional key/legend tasks that `h3` missed, but applying the heavy prompt homogeneously across all 4 attempts still hurt diversity on simple geometric tasks. |
| **`h6`** | **`+` Heterogeneous 2× Fast + 2× Deep Ensemble + $D_8$/Color Equivariance Audit** | **YES (`+3.20 pp` over `h3`, `+7.78 pp` over `h4`)** | `35.28%` / **`36.53%`** | Pairing **Attempts 0–1 (`lean2prog` fast raw)** with **Attempts 2–3 (`lean2progroles` + `task_summary` + `arc_helpers`)** captures both paradigms without correlation collapse, while host-side $D_8$/color equivariance breaks `1-vs-1` verified ties at zero token cost. |
| **`h7`** | **`+` Native Qwen3.8 XML Tool Protocol + 2-Stage Graceful Think Nudge + Final Rescue** | **HUGE YES (`+10.82 pp` over `h6`)** | **`43.81%`** / **`47.35%`** | Switching from JSON `<tool_call>` to Qwen3.8's native post-trained XML `<function=run_python><parameter=code>` eliminated code-escaping failures, and the 2-stage graceful nudge (`2048` grace tokens before forcing code) stopped mid-thought truncation. |
| **`h8`** | **`+` 2-Wave Evolutionary Near-Miss Code Repair (`SOAR` / Berman) + `<think>` Compaction** | **HUGE YES (`+12.62 pp` over `h6`)** | `41.88%` / **`49.15%`** | Seeding Wave 2 (`Attempts 2–3`) with Wave 1's highest-scoring near-miss `def solve(grid):` + exact failing coordinate diffs rescues `90%–98%` cell-match programs that failed 1 training edge case. |

### 4.2 Pilot 1: Additional Hypotheses Tested (`H1`–`H6` across `all_120`, `dev_60`, `hard_25`)
From [`pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md):
* **Non-Destructive Self-Audit (`freeze_first` / `primary_first`)**: **YES (`+1.20%` Pair Pass@2 at 2 attempts, zero compute)**. Post-verification self-audit turns occasionally "fix" imaginary bugs and break a working program; locking the first `train_pass=True` prediction as primary prevents self-sabotage.
* **Outlier Filter + Occam's Razor Tie-Breaking (`-min_output_tokens`, `-min_turns`)**: **YES (`+2.54%` Pair Pass@1, `+1.28%` Pair Pass@2 on `EXP-10C`, zero compute)**. Among multiple `train_pass=True` programs tied `1-1-1-1`, filtering out outlier grids (`peer_sim < 0.5 * max_peer_sim`) and preferring the shortest/earliest-converging trajectory recovers 6 tied test pairs.
* **Multimodal 2D RGB Grid Images (`EXP-13` / `EXP-14`)**: **MIXED for Qwen, STRONG YES for Gemma-4**. On Qwen3.8, standalone image+tools slightly trails text+tools (`40.72%` vs `44.91%` Pair Pass@2) because visual inspection tempts Qwen to hardcode pixel offsets, **but** solves 19 complementary geometric puzzles that lift the combined text+image pool (`EXP-18` / `EXP-19`) to **`52.69%`–`57.49%` Pair Pass@2**. On Gemma-4 31B, adding RGB images + `scipy` + 1 transposed attempt boosted Task Pass@2 from **`22.1%` $\to$ `28.7%` (`+6.6 pp`)**.
* **Test-Time Augmentation (`D8` Rotation/Flip `A3-tta4` & Color Permutation `A3-col4`) vs. Kitchen-Sink Stacking (`A4`)**: **MIXED**. Applying `D8` or color TTA individually unlocks `hard_25` (`0.00%` $\to$ `2.94%–5.88%`), **but** stacking `D8 + Color TTA + Persona Prompts + Struct Summary` simultaneously on the same attempt (`A4` on `hard_25` / `exp5_views16` in Round 2) causes coordinate/color confusion (`-5.28%` in Round 2). **Rule**: Use $D_8$/color equivariance as a **host-side execution audit on `solve(grid)`**, not by mutating the prompt grids simultaneously with persona prompts.

---

## 5. Qwen3.8-27B-FP8 Results Tables

### 5.1 Pilot 2 Controlled Full-Set Ablation (`120` Tasks / `167` Test Pairs — Official Task-Weighted Scoring)
*Source: [`pilot_2/PILOT_2_EXPERIMENTS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/PILOT_2_EXPERIMENTS.md) (`http://xid/302530423` & `http://xid/302763549`).*

| Exp ID | Configuration Summary | Tasks Evaluated | **Official Task Pass@1** | **Official Task Pass@2 (Kaggle)** | Avg Tokens / Task | Compute vs `h1` | Fits Kaggle `4×L4` (`12h`)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`h1_python_tools_base`** | `N=2`, `20` turns, `48k/8k`, JSON tools, `lean2prog` raw grids | `120/120` | `18.33%` (`22.00/120`) | **`19.17%`** (`23.00/120`) | `73.5k` | `1.00×` (`+0%`) | **Yes (`~6.8h`)** |
| **`h2_more_attempts_bo4`** | `h1` + **`N=4` attempts** | `120/120` | `24.31%` (`29.17/120`) | **`26.39%`** (`31.67/120`) | `123.6k` | `1.68×` (`+68%`) | **Yes (`~7.8h` gated)** |
| **`h3_more_max_turns_40`** | `h2` + **`40` turns + `auto_check` coordinate diffs** | `120/120` | `32.22%` (`38.67/120`) | **`33.33%`** (`40.00/120`) | `174.8k` | `2.38×` (`+138%`) | **Yes (`~10.1h` gated)** |
| **`h4_input_desc_and_helpers`** | `h3` + `task_summary` + `arc_helpers` on all 4 attempts *(homogeneous)* | `120/120` | `25.97%` (`31.17/120`) | **`28.75%`** (`34.50/120`) | `182.4k` | `2.48×` (`+148%`) | **Yes (`~10.5h` gated)** |
| **`h5_longer_logical_reasoning`** | `h4` + `lean2progroles` + Falsification (`64k/12k`) on all 4 *(homogeneous)* | `120/120` | `28.33%` (`34.00/120`) | **`31.94%`** (`38.33/120`) | `215.3k` | `2.93×` (`+193%`) | Borderline (`~12.6h` gated) |
| **`h6_adaptive_hybrid_ensemble`** | **`2× fast lean2prog + 2× deep lean2progroles/helpers` + $D_8$/Color Equivariance** | `120/120` | `35.28%` (`42.33/120`) | **`36.53%`** (`43.83/120`) | `165.4k` | `2.25×` (`+125%`) | **Yes (`~9.5h` gated)** |
| **`h7_native_xml_and_grace_nudge`** | `h6` + **Native XML `<function=run_python>` + 2-Stage Graceful Nudge + Rescue** | `113/120` | **`43.81%`** (`49.50/113`) | **`47.35%`** (`53.50/113`) | `228.9k` *(`~122k` gated)* | `+66%` gated | **Yes (`~11.2h` gated)** |
| **`h8_evolutionary_two_wave_repair`** | `h7` + **2-Wave Evolutionary Near-Miss Code Repair + `<think>` Compaction** | `117/120` | `41.88%` (`49.00/117`) | **`49.15%`** (`57.50/117`) | `241.8k` *(`~127k` gated)* | `+72%` gated | **Yes (`~11.6h` gated)** |

---

### 5.2 Pilot 1 Full-Set Runs & Multi-Run Ensembles (`167` Test Pairs — Pair-Weighted Scoring)
*Source: [`pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md) & [`arc_agi_2_gemma4_eval_report.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md).*

| EXP ID | Run Directory / Configuration | Attempts / Pair | Avg Tokens / Attempt | **Pair `Pass@1`** | **Pair `Pass@2`** | Fits Single Kaggle Job? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Exp 1`** | `exp1_no_tools_48k` (`xhigh` thinking, **No Python Tools** baseline) | `2` | `39.0k` | `18.0%` (`30/167`) | `19.8%` (`33/167`) | Yes (`~5.0h`) |
| **`EXP-00`** | `exp2_tools_48k` (Python REPL + `submit_program`, `pooled_latest` voting) | `2` | `20.8k` | `40.12%` (`67/167`) | `43.71%` (`73/167`) | Yes (`2 × 48k`) |
| **`EXP-01`** | `exp7b_n2_48k_ndaudit` (`primary_first` non-destructive self-audit) | `2` | `17.9k` | **`40.72%`** (`68/167`) | **`44.91%`** (`75/167`) | **Yes (`2 × 48k`)** |
| **`EXP-13`** | `exp4_multimodal_tools_pilot_48k` (2D RGB grid images + Python REPL) | `2` | `18.3k` | `37.72%` (`63/167`) | `40.72%` (`68/167`) | Yes (`2 × 48k`) |
| **`EXP-14`** | `exp5_hybrid_audit_48k` (1 Text + 1 Image + visual self-audit) | `2` | `18.7k` | `38.92%` (`65/167`) | `43.11%` (`72/167`) | Yes (`2 × 48k`) |
| **`EXP-10A`** | `exp7a_bo4_96k_ndaudit` (`pooled_latest`, `157` pairs completed) | `4` | `24.1k` | `42.68%` (`67/157`) | `46.50%` (`73/157`) | Borderline (`4 × 48k`) |
| **`EXP-10B`** | `exp7a_bo4_96k_ndaudit` (`primary_first` non-destructive audit) | `4` | `24.1k` | `42.68%` (`67/157`) | `47.13%` (`74/157`) | Borderline (`4 × 48k`) |
| **`EXP-10C`** | `exp7a_bo4_96k_ndaudit` + **Outlier Filter + Fewest Output Tokens (Occam)** | `4` | `24.1k` *(0 new tok)* | **`45.22%`** (`71/157`) | **`48.41%`** (`76/157`) | **Yes w/ Gating** |
| **`EXP-18`** | **4-Run Multi-View Pool** (`EXP-01 + EXP-00 + EXP-14 + EXP-13`) | `8` | `18.9k` | **`46.71%`** (`78/167`) | **`52.69%`** (`88/167`) | Offline (`8 × 48k`) |
| **`EXP-19`** | **5-Run All-Modality Pool** (`EXP-10B + EXP-18`) + **Smart Rating** | `12` | `20.6k` | **`49.70%`** (`83/167`) | **`57.49%`** (`96/167`) | Offline (`12 × 48k`) |

---

### 5.3 Four-Attempt Training Verification & Agreement Breakdown (`EXP-10B`, 157 Pairs / 628 Attempts)
* **Overall `train_pass=True` Rate**: **`48.73%` (`306 / 628` attempts)** pass 100% of training pairs (`50.16%` had at least one `train_pass=True` program during the REPL loop).
* **Test Precision Given `train_pass=True`**: **`54.90%` (`168 / 306`)** exact match on the held-out test grid, compared to **`5.28%` (`17 / 322`)** when an attempt fails training verification (**`10.4×` higher precision**).

| Demo Pass Bucket (`train_pass=True` out of 4 Attempts) | Share of Benchmark | Pair `Pass@1` | Pair `Pass@2` | 4-Attempt Oracle (`bo4`) | Key Scheduling Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`4/4` (`PPPP`)** | `27.39%` (`43/157`) | `65.1%` | `67.4%` | `76.7%` (`33/43`) | When Wave 1 is `PP` with identical test grid (`2/2` vote), accuracy is **`100%` (`7/7`)** $\to$ **Early-stop at `N=2`!** |
| **`3/4` (`PPPF`, etc.)** | `15.92%` (`25/157`) | `56.0%` | `64.0%` | `76.0%` (`19/25`) | Majority vote reliably outvotes the 1 failed/overfit trajectory. |
| **`2/4` (`PPFF`, etc.)** | `10.83%` (`17/157`) | `70.6%` | **`82.4%`** | **`82.4%`** (`14/17`) | Highest `Pass@2` capture rate because the 2 verified programs go directly into `attempt_1` and `attempt_2`. |
| **`1/4` (`PFFF`, etc.)** | `15.92%` (`25/157`) | `44.0%` | `44.0%` | `44.0%` (`11/25`) | Single verified program placed in `attempt_1`; Wave 2 evolutionary repair needed for `attempt_2`. |
| **`0/4` (`FFFF`)** | `29.94%` (`47/157`) | `4.3%` | `6.4%` | `6.4%` (`3/47`) | Target of `h8` Wave 2 near-miss code repair and SFT data generation. |

---

## 6. Gemma-4 (`31B Dense` & `26B MoE`) Results Table

*Evaluated on the full ARC-AGI-2 public evaluation set (`120` tasks, `167` test pairs) via AI Studio / OnePlatform API. Source: [`pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md).*

| Run ID | Model | Configuration & Tools | Pairs Done | **Pair `Pass@1` / `Pass@2`** | **Official Task `Pass@1` / `Pass@2`** | `train_pass` Rate | Avg Output Tokens / Att | Speed (`tok/s`/stream) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | **Gemma-4 26B MoE** (`4B` active) | **High Thinking, No Tools** (`N=2`, `32k` cap) | `167/167` | `1.2%` / `1.8%` (`3/167`) | `1.1%` / **`1.9%`** | — | `26.1k` | `47` |
| **Run 2** | **Gemma-4 31B Dense** | **Minimal Thinking, No Tools** (`N=2`, `16k` cap) | `167/167` | `0.0%` / `0.6%` (`1/167`) | `0.0%` / **`0.3%`** | — | `2.3k` | `34` |
| **Run 3** | **Gemma-4 31B Dense** | **High Thinking, No Tools** (`N=2`, `32k` cap) | `167/167` | `6.0%` / `8.4%` (`14/167`) | `6.5%` / **`9.3%`** | — | `13.4k` | `34` |
| **Run 4** | **Gemma-4 31B Dense** | **High Thinking + Continuation Turns** (`temp=1.0`, `N=2`, `196k` cap) | `167/167` | `7.8%` / `10.8%` (`18/167`) | `7.5%` / **`11.7%`** | — | `13.1k` | `33` |
| **Run 5** | **Gemma-4 31B Dense** | **Python REPL + Train Check** (Writes `solve()`, checked on `train`, `N=2`, $\le 5$ turns) | `167/167` | `16.2%` / `21.6%` (`36/167`) | `16.8%` / **`22.1%`** | `28%` | `33.0k` | `34` |
| **Run 6** | **Gemma-4 31B Dense** | **Python REPL + Prompt Tricks** (Color-role rules + perception summary + ASCII diffs, `N=2`) | `167/167` | `13.2%` / `16.2%` (`27/167`) | `12.1%` / **`15.8%`** | `31%` | `32.0k` | `35` |
| **Run 8** | **Gemma-4 31B Dense** | **Multimodal RGB Image + Python REPL + `scipy` + 3 Attempts** (1 transposed attempt, $\le 7$ turns) | `150/167` | **`25.3%` / `27.3%`** (`41/150`) | **`26.3%` / `28.5%`** *(28.7% @ 125)* | **`34%`** | `31.2k` | `13` |

### Key Takeaways from Gemma-4 vs. Qwen3.8-27B
1. **Active Parameter Capacity Matters for 2D Spatial Working Memory**: `Gemma-4 26B MoE` (`4B` active parameters) scored only `1.9%` Task Pass@2 despite generating `26k` thinking tokens per attempt, whereas `Gemma-4 31B Dense` scored `9.3%–11.7%` without tools and **`22.1%–28.7%` with Python REPL**.
2. **Thinking Horizon Ceiling (`~11k` vs. `40k+` tokens)**: Even when given a `196k` token budget (Run 4), `Gemma-4 31B Dense` naturally terminates its `<think>` block around `~11k–13k` tokens and writes a monolithic `def solve(grid):` on Turn 1 without exploratory print scripts. In contrast, `Qwen3.8-27B` sustainingly debugs across `15–40` interactive turns (`35k–60k` tokens).
3. **Opposite Sensitivity to Multimodal RGB Images**: Adding 2D RGB grid images + `scipy.ndimage` + a transposed attempt (**Run 8**) was **Gemma-4's single largest upgrade (`+9.5 pp` Task Pass@1, `+6.4 pp` Task Pass@2 over text-only REPL Run 5)**, compensating for its shorter text-reasoning horizon with strong visual perception.

---

## 7. What Worked vs. What Didn't Work (Synthesis & Engineering Rules)

### What Worked (High-Leverage Wins)
1. **Interactive Python Program Synthesis + Execution Verification (`+17 pp` to `+30 pp` across both Qwen & Gemma)**:
   * Replacing direct grid transduction (`19.8%` on Qwen, `10.8%` on Gemma-4) with an iterative `run_python` REPL that verifies `def solve(grid):` on `train` pairs is foundational.
2. **Native XML Tool Protocol + 2-Stage Graceful `<think>` Nudge (`h6 -> h7`: `+10.82 pp` Task Pass@2)**:
   * Matching the model's exact post-training tool syntax (`<function=run_python><parameter=code>...</parameter></function>` instead of JSON-escaped strings) and giving `2,048` grace tokens to conclude a long `<think>` block before forcing a tool call raised `train_pass` pair coverage from `46.7%` $\to$ **`68.3%`**.
3. **40-Turn Repair Loop with Automatic Per-Cell Coordinate Diffs (`h2 -> h3`: `+6.94 pp` Task Pass@2)**:
   * `22.3%` of attempts in 16/20-turn baselines were cut off mid-repair with `90%–96%` cell accuracy. Raising `max_turns` to `40` and auto-appending `(r, c): expected X, got Y` tables lets the model fix boundary/off-by-one bugs.
4. **Two-Wave Evolutionary Near-Miss Code Repair (`h7 -> h8`: `+1.80 pp` Task Pass@2 to `49.15%`)**:
   * Seeding Wave 2 (`Attempts 2–3`) with Wave 1's highest-fitness partial `def solve(grid):` code and failing diffs rescues near-miss trajectories instead of restarting from scratch.
5. **Heterogeneous Prompt Diversity Across Attempts (`h4/h5 -> h6`: `+4.59 to +7.78 pp` Task Pass@2)**:
   * Pairing **2 fast raw attempts (`lean2prog`)** with **2 deep role-falsification + structural-summary attempts (`lean2progroles` + `task_summary` + `arc_helpers`)** outperforms running either prompt homogeneously on all 4 attempts.
6. **Zero-Compute Selection & Audit Upgrades (`+2.5% to +3.8%` Pass@1/Pass@2 at `0` extra tokens)**:
   * **First-Verified Lock (`lock_first_verified_pred=True`)**: Never let later self-audit turns overwrite the first `train_pass=True` prediction.
   * **Host-Side $D_8$ & Color-Swap Equivariance Audit**: Testing `solve(T(x)) == T(solve(x))` on `train` breaks `1-vs-1` verified ties in favor of generalizable rules over coordinate-hardcoded shortcuts.
   * **Outlier Filter + Occam's Razor (`-min_gen_tokens`, `-min_turn`, AST constant lint)**: Among tied `train_pass=True` candidates, shorter, earlier-converging programs generalize significantly better to `test`.
7. **2-Wave Gated Early-Stopping for Kaggle `4×L4` (`-46%` tokens, fits `12h` in `~9.5h–11.6h`)**:
   * Stopping at `N=2` whenever both Wave 1 attempts verify on `train` and agree on the test grid (`95%–100%` precision) saves nearly half the compute budget for hard tasks.

### What Didn't Work (Anti-Patterns to Avoid)
1. **Homogeneous Prompt Engineering Across All $N$ Attempts (`h4` in Pilot 2, `Run 6` in Gemma-4)**:
   * Feeding the exact same pre-computed connected-component summary and color-role instructions into *every* attempt collapses ensemble diversity and increases training-pair overfitting (`h3 -> h4` dropped `33.33% -> 28.75%`; Gemma `Run 5 -> Run 6` dropped `22.1% -> 15.8%`).
2. **Stacking $D_8$ Rotation/Flip + Color Permutation + Persona Prompts on the Same Attempt (`A4` / `exp5_views16`)**:
   * Mutating the input grid geometry/colors while simultaneously injecting complex persona instructions confuses coordinate reasoning (`-5.28%` in Round 2).
3. **Destructive Post-Submit Self-Audit (`EXP-00` vs `EXP-01`, `Exp 6`)**:
   * Asking the model to "audit and re-submit" a working `train_pass=True` program without locking the original prediction causes the model to hallucinate edge cases and break working code.
4. **Single-Attempt-Per-Modality Mixing at `N=2` (`Exp 5`: 1 Text + 1 Image)**:
   * At `N=2`, mixing 1 Text attempt and 1 Image attempt leaves `1-vs-1` ties between a correct text program and an overfitted image program with no tie-breaker; multi-view ensembles require either $N \ge 4$ or an equivariance/Occam selector.

---

## 8. SFT Data & 3-Level Verified Trace Generation Summary

Full details are documented in [`ARC_AGI2_SFT_DATA_CATALOG.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/ARC_AGI2_SFT_DATA_CATALOG.md):
* **Standardized Corpus (`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/datasets/`)**: `26` puzzle datasets (**`200,052` tasks**), deduplicated and filtered with a strict zero-leakage blacklist of all `120` official `ARC-AGI-2` eval tasks into **`115,385` unique eligible training tasks across `2,914` families** (`canonical_manifest.jsonl`).
* **3-Level Verification (`train` + held-out `test` + up to `25` withheld generator pairs)**: Recovered `146,532` verified extra pairs across `6,000` NVARC 30-pair tasks and `28,115` extra pairs across `2,884` ARC-GEN 250-pair tasks. Screening proved that **`38.2%` of NVARC programs that pass both `train` and `test` fail on the 25 withheld generator pairs**, making Level-3 filtering essential for clean SFT data.
* **Generated SFT Trajectories (`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/`)**:
  * **`1,175` 3-level verified `Qwen3.8-27B-FP8` trajectories** (`verified_success_manifest.jsonl`, `47.8 MB`) from the 1,000-puzzle empirical screening run.
  * **`1,122` 3-level verified `DeepSeek-V4.1-Flash-1017` full-precision (`BF16 -> FP32`) trajectories** (`deepseek_v41_traces_top10k_ds41_fp_top10k/verified_sft_trajectories_merged.jsonl`, `156.5 MB`) across the **Top 10,000 highest-learning-signal frontier candidates** (`top_10k_deepseek_candidates.jsonl`).
  * **`2,497` failed/near-miss trajectories with exact coordinate diffs** (`failed_trajectories_for_repair.jsonl` + shard failure logs) ready for Wave-2 repair SFT.
