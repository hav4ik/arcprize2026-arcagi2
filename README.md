# ARC-AGI-2: From 19.2% to 49.2% Task Pass@2 with Agentic Python Synthesis

How far can a single open-weights 27B model go on **ARC-AGI-2** (`120` public evaluation tasks, `167` test pairs) when turned into a disciplined, self-verifying Python programmer?

Through a series of controlled experiments (`pilot_1` & `pilot_2`), we scaled **`Qwen3.8-27B-FP8`** from **`19.17%` $\to$ `49.15%` Official Task Pass@2** in a single 4-attempt run (**`57.49%` Pair Pass@2** across our multi-view ensemble), while keeping the full pipeline inside Kaggle's **`4×L4` (`96 GB` VRAM, `12h`)** compute envelope via **2-Wave Gated Early Stopping**.

---

## Headline Results (`Qwen3.8-27B-FP8` on All 120 Tasks)

Each row in `pilot_2` locks the exact same model and sampling parameters (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `131k` context) and changes **one controlled axis at a time**:

| Stage | What We Changed | Attempts | Turns | **Task Pass@1** | **Task Pass@2 (Kaggle)** | Fits `4×L4` (`12h`)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`h1` Baseline** | Basic Python REPL (`lean2prog` raw grids, JSON tool calls) | `2` | `20` | `18.3%` | **`19.2%`** (`23.0/120`) | Yes (`~6.8h`) |
| **`h2` + Attempts** | Double candidate trajectories (`N=2 -> N=4`) + majority vote | `4` | `20` | `24.3%` | **`26.4%`** (`31.7/120`) | Yes (`~7.8h` gated) |
| **`h3` + 40 Turns** | **40-turn loop + auto `check(solve)` cell-coordinate diffs** | `4` | `40` | `32.2%` | **`33.3%`** (`40.0/120`) | Yes (`~10.1h` gated) |
| **`h4` Homogeneous DSL** | Pre-computed CC/symmetry summary + `arc_helpers` on *all 4* | `4` | `40` | `26.0%` | **`28.8%`** *(hurt diversity)* | Yes (`~10.5h` gated) |
| **`h5` Homogeneous Roles**| Role-based falsification (`lean2progroles`, `64k` cap) on *all 4* | `4` | `40` | `28.3%` | **`31.9%`** (`38.3/120`) | Borderline |
| **`h6` Hybrid Ensemble** | **2× Fast Raw + 2× Deep Role/DSL + $D_8$/Color Equivariance** | `4` | `40` | `35.3%` | **`36.5%`** (`43.8/120`) | Yes (`~9.5h` gated) |
| **`h7` Native XML** | **Native XML `<function=run_python>` + 2-Stage Graceful Nudge** | `4` | `40` | **`43.8%`** | **`47.4%`** (`53.5/113`) | Yes (`~11.2h` gated) |
| **`h8` 2-Wave Repair** | **Wave-2 Evolutionary Near-Miss Code Repair + `<think>` Compaction** | `4` | `40` | `41.9%` | **`49.2%`** (`57.5/117`) | **Yes (`~11.6h` gated)** |
| **`EXP-19` Multi-View** | Offline 12-attempt Text + Multimodal RGB Pool + Smart Rating | `12` | `16–40`| **`49.7%`** *(pair)* | **`57.5%`** *(96/167 pairs)* | Offline (`12×48k`) |

---

## Gemma-4 (`31B Dense` & `26B MoE`) vs. Qwen3.8-27B

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

## 5 Breakthrough Lessons (What Actually Worked)

1. **Speak the Model's Native Tool Format (`+10.8 pp` in `h7`)**:
   `Qwen3.8-27B` was post-trained with XML tool tags (`<function=run_python><parameter=code>...</parameter></function>`). Switching from JSON-escaped strings to raw XML code blocks—and adding a `2,048`-token graceful nudge before cutting off long `<think>` blocks—jumped `train`-verified pair coverage from `46.7%` $\to$ **`68.3%`**.
2. **40-Turn Iterative Debugging with Exact Coordinate Diffs (`+6.9 pp` in `h3`)**:
   In 16–20 turn baselines, `22%` of attempts were cut off mid-repair at `90%–96%` cell accuracy. Raising `max_turns=40` and auto-returning `(r, c): got X, expected Y` diffs lets the model fix off-by-one boundary bugs.
3. **Never Homogenize Your Prompts Across Attempts (`h4` vs. `h6`, `+7.8 pp`)**:
   Giving all 4 attempts the same pre-computed connected-component summary (`h4`) collapsed trajectory diversity (`33.3% -> 28.8%`). Pairing **2 fast raw attempts (`lean2prog`)** with **2 deep role-falsification + DSL attempts (`lean2progroles`)** in `h6` beat both homogeneous setups (`36.5%`).
4. **Wave-2 Evolutionary Repair Beats Restarting from Scratch (`+1.8 pp` in `h8` to `49.2%`)**:
   When Wave 1 (`Attempts 0–1`) fails to verify on `train`, seeding Wave 2 (`Attempts 2–3`) with Wave 1's highest-scoring near-miss `def solve(grid):` code and its exact failing diffs rescues near-miss solutions.
5. **Zero-Compute Selection & 2-Wave Gating Fit `N=4` into Kaggle's `4×L4` (`~9.5h–11.6h`)**:
   * **Early-Stop Gate**: When 2 attempts pass 100% of `train` pairs and agree on the test grid (`verified_votes == 2`), test accuracy is **`95%–100%`**. Stopping immediately at `N=2` on those pairs cuts total token usage by **`~46%`**.
   * **Zero-Token Tie-Breakers**: Locking the first `train_pass=True` program (`lock_first_verified_pred`), auditing $D_8$ rotation/color-swap equivariance on `train`, and preferring shorter programs (**Occam's Razor**) adds **`+2.5% to +3.8%` Pass@1/Pass@2** at zero compute cost.

---

## SFT Data & 3-Level Verified Trace Pipeline

Full catalog in [`ARC_AGI2_SFT_DATA_CATALOG.md`](ARC_AGI2_SFT_DATA_CATALOG.md):
* **`200,052` Standardized Tasks (`26` datasets)** deduplicated and filtered (with all `120` official `ARC-AGI-2` eval tasks strictly blacklisted) into **`115,385` unique training puzzles across `2,914` families**.
* **3-Level Verification (`train` + held-out `test` + up to `25` withheld generator pairs)**: Screening `1,000` puzzles proved that **`38.2%` of NVARC programs that pass both `train` and `test` still fail on 25 withheld generator pairs**—making Level-3 verification essential to prevent SFT shortcut training.
* **Generated SFT Trajectories**: **`2,297` strictly 3-level verified multi-turn Python-REPL trajectories** (`1,175` from `Qwen3.8-27B-FP8` + `1,122` from full-precision `DeepSeek-V4.1-Flash-1017` on our Top-10K frontier queue) plus **`2,497` near-miss trajectories** for code-repair SFT.

---

## Repository Map

* [`pilot_2/`](pilot_2/) — Clean, unit-tested TPU/GPU vLLM harness (`pilot2_eval.py`, `prompts_and_selector.py`, `sandbox_verifier.py`, `arc_dsl_and_summary.py`) and full `h1`–`h8` report ([`PILOT_2_EXPERIMENTS.md`](pilot_2/PILOT_2_EXPERIMENTS.md)).
* [`pilot_1/`](pilot_1/) — Initial `EXP-00`–`EXP-19` exploration ([`ARC_AGI2_EXPERIMENTS_FINDINGS.md`](pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md)), Gemma-4 API benchmark ([`arc_agi_2_gemma4_eval_report.md`](pilot_1/setB/results_gemma_api/arc_agi_2_gemma4_eval_report.md)), and `DeepSeek-V4.1-Flash-1017` trace generator.
* [`ARC_AGI2_SFT_DATA_CATALOG.md`](ARC_AGI2_SFT_DATA_CATALOG.md) — 26-dataset SFT catalog, 3-level verification stats, and Top-10K candidate selection breakdown.
* [`arc2_qwen38_tools_v3_ensemble.ipynb`](arc2_qwen38_tools_v3_ensemble.ipynb) · [`arc2-qwen38-e032-sft-ensemble4.ipynb`](arc2-qwen38-e032-sft-ensemble4.ipynb) · [`cell21_patch.py`](cell21_patch.py) — Kaggle `4×L4` notebooks and 2-Wave Gated scheduler.
