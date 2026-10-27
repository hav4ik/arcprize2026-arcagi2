# Qwen3.8-27B-FP8 SFT Evaluation Mini-Report (Post-Harness Fix)

> **Yes — all evaluations in this report use Block-FP8 (`FP8 E4M3`, `128×128` block scales) model weights.**
> Both Full-Parameter SFT (`26.9B` trainable params) and LoRA SFT (`r=64, α=128`, `435M` trainable params) were trained in mixed `BF16/FP32`, merged in exact `FP32`, and quantized back to **Block-FP8 (~27 GB)** so they run at full speed inside the Kaggle hardware/time limits.

---

## 1. Results Summary: Baseline vs. LoRA SFT vs. Full SFT (All `N=2` Attempts, `FP8` Weights)

All runs below are evaluated on the **full 120 official ARC-AGI-2 evaluation tasks (167 test pairs)** using **2 attempts per puzzle (`N=2`)**.
- **Official Kaggle Score (120 Tasks)**: Each of the 120 tasks is weighted equally (`1.0` point per task, averaged across its test pairs).
- **Pair-Level Score (167 Pairs)**: Raw fraction of the 167 test pairs solved.
- **Median Tokens / Attempt**: Shown both across **all attempts** and across **solved attempts** (`Solved: ...`).

### A. Main Comparison (`60k` Token Cap vs. Comparable Baselines)

| Model / Checkpoint (`FP8` Weights) | Setup Summary | Official Kaggle Pass@1 (120 Tasks) | Official Kaggle Pass@2 (120 Tasks) | Pair-Level Pass@1 (167 Pairs) | Pair-Level Pass@2 (167 Pairs) | Median Tokens / Attempt (Solved Median) | Total Eval Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Comparable Baseline (`h1`, Base FP8)** | Pure Qwen3.8-27B-FP8, JSON Python tool calls (`48k` cap) | `18.33%` (`22.0/120`) | **`19.17%`** (`23.0/120`) | `16.77%` (`28/167`) | `17.96%` (`30/167`) | `25,015` (`Solved: 28,649`) | `8.82M` |
| **2. Collaborator Base (`E020 v4`, Base FP8)** | Pure Qwen3.8-27B-FP8, XML tools + `xhigh` tier (`48k` cap) | `34.17%` (`41.0/120`) | **`37.08%`** (`44.5/120`) | — | — | `~31,400` | `10.50M` |
| **3. Collaborator Base (`E031/E032`, Base FP8)** | Pure Qwen3.8-27B-FP8, `E031` prompt + `xhigh` tier (`60k` cap) | `42.92%` (`51.5/120`) | **`49.17%`** (`59.0/120`) | — | — | `~38,900` | `13.00M` |
| **4. LoRA SFT Step 150 (`r=64`, FP8)** | Fixed `E031` harness (`60k` cap) | `37.92%` (`45.5/120`) | **`43.61%`** (`52.3/120`) | `37.72%` (`63/167`) | `43.11%` (`72/167`) | `59,842` (`Solved: 25,490`) | `15.59M` |
| **5. LoRA SFT Step 191 Final (`r=64`, FP8)** | Fixed `E031` harness (`60k` cap) | `42.08%` (`50.5/120`) | **`47.08%`** (`56.5/120`) | `40.12%` (`67/167`) | `44.91%` (`75/167`) | `59,652` (`Solved: 21,301`) | `14.88M` |
| **6. Full SFT Step 192 Final (`26.9B`, FP8)** | Fixed `E031` harness (`60k` cap) | `44.17%` (`53.0/120`) | **`50.00%`** (`60.0/120`) | `43.11%` (`72/167`) | `49.10%` (`82/167`) | `46,879` (`Solved: 21,376`) | `13.58M` |
| **7. Full SFT Step 100 (`26.9B`, FP8) ★ BEST** | Fixed `E031` harness (`60k` cap) | **`50.00%`** (`60.0/120`) | **`55.00%`** (`66.0/120`) | **`47.31%`** (`79/167`) | **`51.50%`** (`86/167`) | `53,478` (`Solved: 22,818`) | `14.41M` |

---

### B. Fast Kaggle `<6h` Cap (`42k` Token Cap, `N=2` Attempts, `FP8` Weights)

Because solved SFT attempts finish in a median of only **`18k–22k` tokens**, capping each attempt at **`42,000` tokens** keeps total token usage around **`10.9M–11.6M` tokens** (`~4.5–4.9 hours` on 4×L4 GPUs, well under the 6-hour Kaggle limit of `13.6M` tokens):

| Model / Checkpoint (`FP8` Weights) | Token Cap | Official Kaggle Pass@1 (120 Tasks) | Official Kaggle Pass@2 (120 Tasks) | Pair-Level Pass@1 (167 Pairs) | Pair-Level Pass@2 (167 Pairs) | Median Tokens / Attempt (Solved Median) | Total Eval Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full SFT Step 192 Final (`26.9B`, FP8) ★ Best `<6h`** | `42k` | **`44.03%`** (`52.8/120`) | **`44.86%`** (`53.8/120`) | **`43.71%`** (`73/167`) | **`44.91%`** (`75/167`) | `41,744` (`Solved: 18,002`) | **`10.93M`** (`~4.8h`) |
| **LoRA SFT Step 191 Final (`r=64`, FP8)** | `42k` | `41.67%` (`50.0/120`) | **`43.19%`** (`51.8/120`) | `40.12%` (`67/167`) | `42.51%` (`71/167`) | `41,886` (`Solved: 21,689`) | **`11.21M`** (`~4.9h`) |
| **Full SFT Step 100 (`26.9B`, FP8)** | `42k` | `38.75%` (`46.5/120`) | **`42.92%`** (`51.5/120`) | `36.53%` (`61/167`) | `40.12%` (`67/167`) | `42,000` (`Solved: 20,058`) | **`11.17M`** (`~4.9h`) |
| **LoRA SFT Step 150 (`r=64`, FP8)** | `42k` | `31.25%` (`37.5/120`) | **`33.75%`** (`40.5/120`) | `30.54%` (`51/167`) | `32.93%` (`55/167`) | `42,000` (`Solved: 17,562`) | **`11.61M`** (`~5.1h`) |

---

## 2. Before vs. After the Harness Fixes: Why Full SFT Jumped +14.2%

Before these fixes, Full SFT Step 100 scored **`40.76%`** Official Kaggle Pass@2 (`36.55%` Pass@1) and looked *worse* than LoRA SFT (`46.94%`), even though Full SFT had a lower validation loss (`0.7549` vs. `0.7627`).

After fixing the evaluation harness, **Full SFT Step 100 jumped to `55.00%` Official Kaggle Pass@2 (`50.00%` Pass@1) and `51.50%` Pair-Level Pass@2 (`47.31%` Pass@1)** — beating our comparable baseline (`19.17%`) by **+35.8 percentage points**, beating collaborator Vu's `E020 v4` (`37.08%`) by **+17.9 percentage points**, and beating Vu's best `E031/E032` base model (`49.17%`) by **+5.8 percentage points**.

---

## 3. What Was Fixed (Plain-English Breakdown)

Here are the **5 specific bugs** we fixed in the evaluation harness and why they mattered so much:

### Fix 1: Restoring Qwen's Built-In `tier=xhigh` System Header (Removing Prompt Mismatch)
- **What was broken**: Qwen3.8-27B was pre-trained by Alibaba with a specific magic sentence at the very top of its system prompt: `"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions..."`. In our earlier evals, we accidentally replaced that sentence with a custom string (`"Reasoning Effort: Minimal"`).
- **What we fixed**: We restored Qwen's exact pre-trained `tier=xhigh` sentence at the top of the system prompt while keeping the brief-code instruction (`"Reasoning Effort: Minimal. Keep your thinking short and terse..."`) at the bottom of the tool block — matching both Qwen's pre-training and our SFT training format.

### Fix 2: Capping Single-Turn Thinking at 8k Tokens + 2-Stage Graceful XML Nudge
- **What was broken**: Previously, the `E031` eval allowed the model to think for up to `32,000` tokens in a single turn without running any Python code. On hard puzzles, the model would get stuck over-thinking in `<think>...</think>` and burn half its total budget before testing a single line of Python.
- **What we fixed**: We capped each turn's thinking at `8,192` tokens and enabled a **2-stage graceful nudge**:
  1. **Stage 1 (at 8k tokens)**: If the model is still inside `<think>`, we append `" ... Wrapping up my reasoning cleanly so I can test in Python.\n</think>\n\n"` and give it `2,048` grace tokens to write its `<tool_call>` naturally.
  2. **Stage 2 (if it still doesn't call a tool)**: We force-close `</think>` and open `<tool_call>\n<function=run_python>\n<parameter=code>\n` so the model immediately writes executable Python code instead of losing the turn.

### Fix 3: Fixing the Mid-`<tool_call>` Code Truncation Bug
- **What was broken**: In the old loop, when a turn hit the token limit (`finish_reason == "length"`), the harness checked `if "<tool_call>" not in text:`. If the model finished `</think>` near token 8,000 and was **halfway through writing Python code** inside `<tool_call>` when token 8,192 hit, `"<tool_call>" not in text` was `False`! So the harness immediately sent the half-written Python script to the sandbox, causing an instant `SyntaxError` and wasting a turn.
- **What we fixed**: We changed the check to verify that the XML tool call is **actually finished** (`"</tool_call>" in text or "</function>" in text`). If the model is cut off mid-code, we give it `2,048` grace tokens to finish writing the rest of its Python script and close `</tool_call>`.

### Fix 4: One-Time Rejection of Unverified Submissions (`verify_submit`)
- **What was broken**: Previously, if the model called `submit_answer(solve(test_input))` with a buggy `solve()` function that didn't even work on the training examples, the harness accepted the grid immediately and ended the episode — throwing away the remaining token budget.
- **What we fixed**: When `submit_answer` is called, the harness automatically tests the submitted expression/function against all training pairs. If it fails on any training pair, the harness **rejects the submission once** (`"[submission rejected: your rule only matched X/Y training pairs...]"`) and lets the model keep debugging in Python (while still saving that grid as a backup in case time runs out).

### Fix 5: Salvaging Working Functions Before Forcing a Last-Second Guess (`force_submit_last`)
- **What was broken**: When an attempt ran out of budget without calling `submit_answer`, the old harness first forced the model to write a hasty 1-line `submit_answer` guess, which overwrote `submitted_answer_grid` and prevented the session-salvage fallback from running.
- **What we fixed**: When budget expires, we **first** inspect the live Python memory for `answer`, `solve(test_input)`, or any helper function (`fn(grid)`) that reproduces the training outputs (`salvage_answer`), and only ask the model for a forced last-second guess if no working function exists in Python memory.
