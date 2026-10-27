# ARC-AGI-2 Experiments Report: Qwen3.8-27B & Gemma-4

Evaluated on the full **ARC-AGI-2 public evaluation set** (`120 tasks`, `167 test grids`, 2 predictions allowed per grid).

---

## 1. Results Summary

### Part A: Qwen3.8-27B-FP8 (Run on TPUv7 in FP8 via vLLM)
*Each run uses 4 shards of `4x TPUv7` (`ghostfish=2x2x1`) with FP8 weights (`27.4 GB`) and vLLM prefix caching.*

| Exp | Short Description | Completed | Kaggle `pass@1` / `pass@2` | ARC Prize `pass@1` / `pass@2` | Train Pass Rate | Avg Tokens / Attempt | Speed (`tok/s`) | Fits on Kaggle (`4xL4`, 12h)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1** | **No Tools Baseline** (`xhigh` thinking, single turn, `N=2`, `48k` cap) | **Yes** (`167/167`) | **18.0% / 19.8%** | **19.3% / 20.4%** | — | `39k` | `1,018` | **Yes** (~5h) |
| **Exp 2** | **Python Tools Baseline** (Python REPL + `submit_program` train check + 8k turn cap, `N=2`, `48k` cap) | **Yes** (`167/167`) | **32.3% / 37.1%** | **32.6% / 36.7%** | `52%` | `35k` | `422` | **Yes** (~5.2h) |
| **Exp 3** | **Best-of-8 Tools Scaling** (Same REPL + majority vote across 8 attempts + early stop, `N=8`, `96k` cap) | **No** (`98/167` running) | **50.0% / 54.1%** | **51.2% / 55.8%** | `70%` | `45k` | `413` | **No** (~24h; fits `N=3` in ~8.5h) |
| **Exp 4** | **Multimodal Image + Tools** (RGB task image + REPL + rotated/flipped 2nd attempt, `N=2`, `48k` cap) | **Yes** (`167/167`) | **27.5% / 35.3%** | **30.7% / 38.3%** | `53%` | `34k` | `329` | **Yes** (~5.8h) |
| **Exp 5** | **Hybrid Text + Image + Audit** (Attempt 1 Text+REPL, Attempt 2 Image+REPL + 1-turn post-submit check, `N=2`, `48k` cap) | **No** (`126/167` running) | **33.3% / 36.5%** | **33.8% / 37.3%** | `55%` | `35k` | `412` | **Yes** (~5.4h) |
| **Exp 6** | **Upgraded Tools Harness** (Cross-pair `solve()` sharing + coordinate diff + 1-turn audit + TTA, `N=2`, `56k` cap) | **No** (`136/167` running) | **27.9% / 36.0%** | **29.7% / 38.4%** | `54%` | `39k` | `392` | **Yes** (~6.1h) |
| **Union (2+4)** | **Combined Pool of Exp 2 (Text) + Exp 4 (Image)** (`4` attempts total per grid) | **Yes** (`167/167`) | — / **48.5%** | — / **49.6%** | `65%` | `34k` | — | **Yes** (~7.8h w/ early stop) |

---

### Part B: Gemma-4 31B Dense & 26B MoE (Run via AI Studio API)
*Served through the AI Studio API (14 parallel worker streams).*

| Run | Model | Short Description | Completed | Kaggle `pass@1` / `pass@2` | ARC Prize `pass@1` / `pass@2` | Train Pass Rate | Avg Tokens / Attempt | Speed (`tok/s`) | Fits on Kaggle (`4xL4`, 12h)? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1** | **Gemma-4 26B MoE** (`4B` active) | **High Thinking, No Tools** (`N=2`, `32k` cap) | **Yes** (`167/167`) | **1.2% / 1.8%** | **1.1% / 1.9%** | — | `26k` | `47` / stream | **Yes** (~3.2h) |
| **Run 2** | **Gemma-4 31B Dense** | **Minimal Thinking, No Tools** (`N=2`, `16k` cap) | **Yes** (`167/167`) | **0.0% / 0.6%** | **0.0% / 0.3%** | — | `2k` | `34` / stream | **Yes** (~0.5h) |
| **Run 3** | **Gemma-4 31B Dense** | **High Thinking, No Tools** (`N=2`, `32k` cap) | **Yes** (`167/167`) | **6.0% / 8.4%** | **6.5% / 9.3%** | — | `13k` | `34` / stream | **Yes** (~2.0h) |
| **Run 4** | **Gemma-4 31B Dense** | **High Thinking + Continuation Turns** (`temp=1.0`, `N=2`, `196k` cap) | **Yes** (`167/167`) | **7.8% / 10.8%** | **7.5% / 11.7%** | — | `13k` | `33` / stream | **Yes** (~2.0h) |
| **Run 5** | **Gemma-4 31B Dense** | **Python REPL + Train Check** (Writes `solve()`, checked on training pairs, `N=2`, $\le 5$ turns) | **Yes** (`167/167`) | **16.2% / 21.6%** | **16.8% / 22.1%** | `28%` | `33k` | `34` / stream | **Yes** (~5.5h) |
| **Run 6** | **Gemma-4 31B Dense** | **Python REPL + Prompt Tricks** (Added color-role rules, perception summary, ASCII diffs, `N=2`) | **Yes** (`167/167`) | **13.2% / 16.2%** | **12.1% / 15.8%** | `31%` | `32k` | `35` / stream | **Yes** (~5.3h) |
| **Run 8** | **Gemma-4 31B Dense** | **Multimodal Image + REPL + 3 Attempts** (RGB image + `scipy` + 1 transposed attempt, `N=3`, $\le 7$ turns) | **No** (`125/167` running) | **26.4% / 27.2%** | **27.5% / 28.7%** | `34%` | `31k` | `13` / stream | **Borderline** (~8.2h) |

---

## 2. What Worked and Why (Experiment Breakdown)

### Qwen3.8-27B (TPUv7, FP8)
* **Exp 1 (No Tools, `19.8%` `pass@2`)**: Without tools, Qwen thinks for `~45k+` tokens on hard puzzles. In **62% of attempts**, it hit the `48k` token ceiling mid-thought before printing an answer. When it finished under `48k`, it was very accurate (`33%` exact match).
* **Exp 2 (Python REPL, `37.1%` `pass@2` — `+17.3%` jump)**: Capping each thinking step at `8k` tokens and forcing the model to test code in Python (`run_python` + `submit_program`) stopped it from getting cut off. **52% of puzzles** got a `solve()` function that passed all training pairs, and **61% of those** were right on the test grid.
* **Exp 3 (Best-of-8, `54.1%` Kaggle / `55.8%` ARC Prize `pass@2` — `+17.0%` jump)**: Giving the model 8 attempts and `96k` tokens boosted training-pair pass coverage from `52%` to **`70%`**. More importantly, **majority voting across the 8 attempts filtered out overfitted programs**: wrong programs produce different test grids, while the right rule converges to the same answer (`77%` accuracy when voted).
* **Exp 4 (Image + Tools, `35.3%` Kaggle / `38.3%` ARC Prize `pass@2`)**: Passing the puzzle as an RGB image solved **19 new puzzles that Text-only (Exp 2) missed**, lifting task-weighted ARC Prize `pass@2` to `38.3%`. However, `pass@1` dropped (`32.3%` $\to$ `27.5%`) because seeing the image made Qwen visually guess offsets instead of writing careful `numpy` code.
* **Exp 5 (1 Text + 1 Image + Audit, `33.3%` `pass@1` / `36.5%` `pass@2`)**: Splitting the 2 attempts into 1 Text and 1 Image gave our **highest `N=2` `pass@1` (`33.3%`)**, but `pass@2` stayed around `36.5%` (explained in Section 3 below).
* **Exp 6 (Upgraded Tools Harness, `36.0%` Kaggle / `38.4%` ARC Prize `pass@2`)**: Sharing `solve()` across sibling test pairs of the same task and auditing shapes gave our **highest `N=2` ARC Prize `pass@2` (`38.4%`)** and cut wrong-size answers from `30%` to `21%`. However, letting the 1-turn post-submit audit overwrite `attempt_1` hurt `pass@1` (`27.9%`) because the model sometimes broke an already-working program when trying to silence audit warnings.

### Gemma-4 (AI Studio API)
* **Dense 31B vs. MoE 26B (Runs 1–4)**: The MoE (`4B` active parameters) scored `1.8%`—it cannot track 2D coordinates in working memory. Dense 31B scored `8.4%` (`10.8%` with continuation turns), but **Gemma-4 stops thinking at `~11k` tokens on its own**, even when given a `196k` cap.
* **Adding Python REPL (Run 5, `21.6%` `pass@2`)**: Giving Gemma-4 a 5-turn Python REPL doubled its score (`10.8%` $\to$ `21.6%`).
* **Adding Prompt Tricks Hurt (Run 6, `16.2%` `pass@2`)**: Giving Gemma-4 pre-computed perception summaries and cell-level diffs made it overfit the training pairs (`train`-passing attempts went up from `66` to `89`, but test accuracy on those programs dropped from `58%` to `45%`).
* **Adding RGB Image + `scipy` + 3 Attempts (Run 8, `27.2%` Kaggle / `28.7%` ARC Prize)**: Unlike Qwen, **Gemma-4 benefited hugely (`+10.2%` `pass@1`, `+5.6%` `pass@2`) from seeing the RGB image** alongside `scipy.ndimage` and 3 ranked attempts.

---

## 3. Failure Analysis

### 3.1 Why Models Fail on ARC-AGI-2
1. **Overfitting the Training Examples (~40% of training-passing programs fail on the test grid)**:
   - Most ARC puzzles have only 2–4 training examples. Both models frequently write a `solve()` function that passes 100% of the training examples by relying on an accidental coincidence—such as hardcoding a color (`color == 2`) that happened to appear in all training pairs, or assuming an object always moves right.
2. **1-to-4 Pixel Near Misses (10%–18% of all attempts get the right size and $\ge 95\%$ of cells right)**:
   - In both Qwen (`Exp 2–6`) and Gemma (`Runs 5–6`), roughly 1 in 7 attempts finds the exact right rule and grid size, but misses by a few pixels due to an off-by-one slice (`r1` vs `r1 + 1`), ray endpoint, or color-overlap order.
3. **Wrong Output Dimensions on Size-Changing Puzzles (21%–30% of Qwen attempts)**:
   - When the output grid is cropped or resized and the model fails to pass the training check, our fallback returns a grid of the input's size.

### 3.2 Are Failures Similar Across Gemma-4 and Qwen3.8?
* **Same hard puzzles**: Both models fail most often on **large grids (`> 600` cells, up to $30 \times 30$)** and **2-example puzzles** (where rules are easiest to overfit). Almost every puzzle solved by Gemma-4 (`92%`) is also solved by Qwen3.8.
* **Different coding habits**: Gemma-4 immediately writes a complete `def solve(grid):` on Turn 1 without inspecting the data first, and stops thinking at `~11k` tokens. Qwen3.8 writes short inspection scripts across `12–15` turns and continues finding solutions up to `70k+` tokens.

### 3.3 Why Ensembling Image + Tools (`Exp 5`) Didn't Automatically Reach `48.5%`
Pooling all 4 attempts from **Exp 2 (2 Text attempts)** and **Exp 4 (2 Image attempts)** solves **48.5% (`81/167`)** of the benchmark (`22` solved only by Text, `19` solved only by Image). Yet **Exp 5 (`1 Text attempt + 1 Image attempt`)** scored **36.5% `pass@2`**. Three reasons explain why:
1. **Giving up the 2nd attempt per modality**: Single tool runs are noisy. In Exp 2, the 2nd Text attempt added `+5%` accuracy; in Exp 4, the 2nd Image attempt added `+8%`. When Exp 5 used only **1 Text attempt + 1 Image attempt**, losing the 2nd retry inside each modality cancelled out the gain from mixing modalities.
2. **Vision programs overfit more often**: In Exp 5, Text programs that passed `train` were right on `test` **66% of the time**, whereas Image programs that passed `train` were right on `test` only **58% of the time** (so Attempt 0 Text got `31.0%` exact vs `23.8%` for Attempt 1 Image).
3. **No tie-breaker when `N = 2`**: On 11 puzzles in Exp 5, *both* the Text attempt and the Image attempt passed 100% of training examples with different test grids (one right, one overfitted). With only 2 attempts, each grid has 1 vote, so we cannot tell which one is overfitted without a 3rd or 4th attempt to break the tie.

---

## 4. Next Steps & Future Hypotheses

1. **Adaptive 2-Wave `3 Text + 1 Image` Ensemble (`+10% to +13%` expected `pass@2`, fits Kaggle in ~7.8h)**:
   - Run **2 Text+Tools attempts** first. If both pass `train` and produce the exact same test grid (~45% of puzzles, `>85%` accuracy), **stop immediately**. Only if they fail `train` or disagree, run **1 more Text attempt + 1 Image attempt** (4 attempts total) and majority-vote.
2. **Non-Destructive Post-Submit Audit (`+4%` `pass@1`, 0 extra tokens)**:
   - Never let the 1-turn post-submit audit overwrite the first `train`-passing grid. Keep the first `train`-passing program locked as `attempt_1`, and save any post-audit edit as `attempt_2`.
3. **Rotation & Color-Swap Check Inside `submit_program` (`+3% to +5%`, <10ms CPU)**:
   - When `solve()` passes `train`, automatically test if `solve(rot90(x)) == rot90(y)` and `solve(color_swap(x)) == color_swap(y)` on the training pairs. Programs that pass this check cannot have hardcoded coordinates or colors and should be ranked first.
4. **1-Turn Pixel Tie-Breaker for Near Misses (`+2% to +4%`)**:
   - When two `train`-passing attempts output the same grid shape but disagree on just 1–5 pixels (`10%–16%` of attempts), ask the model in 1 short turn to inspect those exact `(row, col)` coordinates.
