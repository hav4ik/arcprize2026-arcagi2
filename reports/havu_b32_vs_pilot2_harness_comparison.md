<!-- go/g3mark-in-g3doc -->
<!-- disableFinding(LINE_OVER_80) -->

# Havu Candidate B32 vs. `pilot_2` (`e031_deepseek_teaching`) Harness Comparison

<!--* freshness: { owner: 'arazdai' reviewed: '2026-10-28' } *-->

Comparison between **Havu's latest best-performing ARC-AGI-2 harness (`Candidate B32`**, `hav4ik/arcprize2026-arcagi2` @ `main` `91ecbcc1`: `kaggle/dsv41_tools/CANDIDATE_B32.md`, `experiments/E055_kaggle_replica/launch_candBf_x2t.sh`, `kaggle/dsv41_tools/solver_dsv41.py`, `evals/run_deepseek_tools.py`, `evals/auto_repair.py`, `evals/train_check.py`, `evals/arc_harness/executor.py`) and our current **`pilot_2` evaluation harness** (`e031_deepseek_teaching` and `e031_deepseek_teaching_bo4` in `experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/`).

[TOC]

## 1. Selected Checkpoint from Last Full SFT Run (`R2-A`)

The last full-parameter SFT run is **`R2-A`** (`xid/304951546`, resumed cleanly to step 486/486 in `xid/305098001` on `Qwen3.8-27B` with curated DeepSeek-teaching trajectories, `bs=32`, `lr=1e-5`). Across all 8 exported FP8 checkpoints evaluated on the full 120-task / 167-pair ARC-AGI-2 evaluation split, **`final` (`step-00486`)** is #1 on every metric:

| Checkpoint | Step | Val Token NLL | `e031_deepseek_teaching` (`N=2`) Official Task Pass@2 | `e031_deepseek_teaching_bo4` (`N=4`) Official Task Pass@1 | `e031_deepseek_teaching_bo4` (`N=4`) Official Task Pass@2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `step-00050` | 50 | 0.7289 | 45.42% | 48.75% | 53.33% |
| `step-00100` | 100 | 0.7014 | 47.08% | 51.25% | 56.25% |
| `step-00150` | 150 | 0.6892 | 48.33% | 52.50% | 57.50% |
| `step-00200` | 200 | 0.6815 | 49.58% | 53.75% | 58.33% |
| `step-00300` | 300 | 0.6748 | 50.83% | 54.58% | 59.58% |
| `step-00400` | 400 | 0.6716 | 51.67% | 55.42% | 60.42% |
| **`final` (`step-00486`)** | **486** | **0.6700** | **52.50% (`63.00/120`)** | **56.25% (`67.50/120`)** | **61.53% (`73.83/120`)** |

- **Selected Checkpoint Path**: `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_full_sft_v2/xid_304951546_qwen38_27b_arc2_r2_curated_bs32_lr1e-5/wu_1/fp8_checkpoints/final`
- **Previous `pilot_2` Eval Reference**: `xid/305119533` (`/cns/lu-d/home/arazdai/kaggle/arc_agi_2/r2_results/r2_curated_bs32_lr1e5_xid304951546/final`)

---

## 2. Executive Summary of the 12 Architectural Differences

While both harnesses share the same core **`lean2sub` prompt + `run_python` / `submit_answer` IPython interface + `minimal` effort system header**, Havu's **Candidate B32** (`E043` + `E046` + `E055` + `A1–A11` + `I5` + `S14`) differs fundamentally in **how rollouts are structured across test pairs, how token history is fed to vLLM, whether thinking is interrupted mid-turn, how trivial identifier typos are repaired at zero token cost, how code execution is isolated, and how `train_pass` is verified post-hoc**:

| # | Dimension | Our `pilot_2` (`e031_deepseek_teaching` / `_bo4`) | Havu's Latest Best (`Candidate B32`) | Impact / Why It Matters |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Multi-Test Rollout Granularity** | **1 rollout per `(task, pair)`** (`167 × N` rollouts); sibling test inputs are invisible to each other | **`ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`** (`120 × N` rollouts): shows all test inputs (`--Test input 0--`, ...) in 1 rollout, preloads `test_inputs`, scales budget & turns by `1 + 0.5*(n-1)` | Model sees cross-test edge cases simultaneously (shapes, colors, counts) and cuts multi-test compute ~29–41% |
| **2** | **Identifier Typo Auto-Repair** | **None**: `NameError` / `AttributeError` / `ImportError` / kwarg `TypeError` returned raw to LLM | **`ARC_AUTO_REPAIR=1` (`auto_repair.py`, Fix I5)**: AST side-effect scan + unique OSA distance $\le 2$ token rewrite + 1 free in-place retry (`[auto-corrected: a -> b]`) | Fixes ~52–61% of identifier/BPE typos at **0 model tokens** and **0 wasted turns** (90–93% match to model's next fix) |
| **3** | **Tokenization & Transcript Compaction** | **String re-tokenization every turn** + premature stop on `"</tool_call>"` + **mid-run `<think>` compaction** (`>56k` chars strips old `<think>` & truncates tool outputs to 1.5k) | **`--token-exact` (`TokensPrompt`) + ZERO compaction**: preserves exact `prompt_ids + sampled_ids` across turns, stops on EOS (`<\|im_end\|>`), resyncs only on rescued/cut turns | Eliminates ~13% BPE boundary drift and prevents mid-rollout KV prefix-cache invalidation |
| **4** | **Per-Turn Thinking Cap & Mid-Thought Injection** | **Hard `8,192`-token cap** + 2-stage synthetic injection inside open `<think>` (`XML_THINK_NUDGE_STAGE1` -> 2,048 grace -> `XML_THINK_FORCE_ACT_STAGE2`) + forced final turn | **Uninterrupted `max_tokens=64,000` per turn**, `--cap-nudge` **OFF**, `--final-mode session` (no forced final turn), seamless continuation on spurious EOS inside `<think>` (up to 4×) | Never derails deep multi-step deductions at 8k tokens or injects out-of-distribution synthetic `<think>` text |
| **5** | **Token Budget, Soft Target & Tool Limits** | `60k` hard / `20k` soft, **`"strong"`** wording (*"submit now unless..."*), `3,500`-char tool output cap, `20s` cell timeout | **`80k` hard / `40k` soft** (scaled on multi-test), **`"gentle"`** wording (*"submit as soon as a rule fits every training pair"*), **`16,000`-char** tool output cap, **`30s`** timeout, `turns_note=5`, `budget_warn=0.75`, context-aware budget (**Fix A9**) | 4.5× more stdout headroom (`16k` vs `3.5k` prevents truncated grid inspections) and avoids premature 20k panic submissions |
| **6** | **`submit_answer` Gate & Hidden Checks** | **`verify_submit=True`** (rejects `submit_answer` once if `train` re-eval fails) + hidden `solve()` check after every `run_python` cell | **`verify_submit=False`**: immediately accepts any valid 1–30×1–30 `0..9` grid (or list of `n` grids); zero hidden cell checks during rollout | Avoids blocking scripts that compute `answer` Stateful/imperatively without a pure `test_input` expression |
| **7** | **Python Sandbox Isolation** | **In-process host `threading.Thread` + `sys.settrace`**; preloads extra globals (`test_in`, `task`, `scipy`, `ndimage`, `Counter`, `combinations`) | **Isolated child process per attempt (`LocalExecutor`, `style="ipython"`)**: `RLIMIT_AS=4096MB`, `RLIMIT_FSIZE=64MB`, `oom_score_adj=1000`, `OMP_NUM_THREADS=1`, `SIGINT` + 5s grace then `SIGKILL` & clean respawn | Stops hung C/NumPy/SciPy calls cleanly without leaking host threads or OOMing the vLLM worker |
| **8** | **Session & `solve()` Harvest Hardening** | Single-shot `answer` check -> list-only `solve(copy.deepcopy(test_input))` -> `force_submit_last` turn | **Fixes A1, A2, A3, A4, A7, A8, A10, A11**: `restore_inputs`, shadow-proof `builtins as _b`, nested ndarray/bool `_plain`, chunked `ANSWER_SLICE` (>16k), per-item partial multi-test recovery, `solve(list)` **then** `solve(np.array)`, `cut_parse`, error-path harvest | Recovers valid predictions from numpy-only `solve(arr)`, `def eval(...)` shadowing, rebound `test_input`, >16k multi-grids, and cut/errored attempts |
| **9** | **Malformed / Plain-Code Rescue** | Missing `<tool_call>` returns a synthetic `<tool_response>` parse error | **`--rescue-code` + `RESCUE_NUDGE` + `--lenient-parse`**: extracts code from broken XML/DSML tags, `"code":` JSON, ` ```python ` fences, or raw code lines into `run_python`; normalizes tool-name typos (`rescue_name`); up to 3 user nudges | Converts formatting slips on long turns directly into productive code execution instead of wasted error turns |
| **10** | **Post-Attempt `train_check`** | Checks `def solve(...)` (or `test_input` expression) only; direct `answer = ...` scripts or `def transform(g):` get `train_pass=False` | **4-path zero-token `train_check.py` (`expr` → `cell` → `fn` → `transcript` fresh replay + Fix A6)**: verifies SHA-1 test reproduction (`ph`) first, then tests expressions, last `answer`-assigning cell, up to 40 1-arg user functions (`list` & `np.array`), and full fresh-session cell replay | Accurately labels `train_pass` for imperative notebooks, custom function names, and multi-cell pipelines |
| **11** | **Candidate Selection & Cross-Attempt Policy** | `_bo4` uses 2-wave evolutionary cross-attempt prompt seeding; selector uses weighted vote + `0.25 * D8/color equivariance` + token-length tie-break | **Independent attempts** (`prior_mode` OFF) + **`train_check.ranked_answers`**: strictly `(train_pass is not True, attempt_idx)` for top-2 distinct grids (padded with `test_input` fallback) | Clean, hyperparameter-free selection driven by the much higher recall of the 4-path `train_check` |
| **12** | **Runtime Loop Guard & Queue Scheduling** | No runtime doom-loop detector; alphabetical pair sharding | **`GuardedStream` thinking gzip monitor** (`1,000`-token windows; 10 consecutive windows at `gzip_ratio > 8.0` inside `<think>` aborts stream -> session `answer` fallback) + **smallest-prompt-first ordering** + all `attempt=0` before `attempt=1` (prioritizing non-`train_pass` tasks) | Terminates pathological repetition loops early without losing session `answer` and maximizes throughput |

---

## 3. Detailed Technical Breakdown

### 3.1 Task-Level Multi-Test Rollouts (`ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, Fix A5)
In ARC-AGI-2 `evaluation`, 120 tasks contain **167 test pairs** (39 tasks have 2–4 test inputs).
- **`pilot_2` (`e031_deepseek_teaching`)**: Splits the 120 tasks into 167 independent `EvalPairSpec` items. On a 3-test-pair task, the model solves each test input in isolation without seeing the other 2 test inputs, often missing structural constraints (e.g., grid sizes, object counts, or color-role generalizations) that are immediately obvious when viewing all test inputs together.
- **Havu `Candidate B32`**: Runs at the **task** level (`120 × N` rollouts):
  - When `len(task["test"]) == n > 1`, `build_prompt_multi` formats `--Test input 0--`, `--Test input 1--`, ..., and appends `MULTI_NOTE`:
    > *"This task has {n} test inputs, shown above in order and preloaded in the session as `test_inputs` (a list of {n} grids; test_input is test_inputs[0]). One rule must work for all of them: check its assumptions on every test input. Submit a list with one output grid per test input, in order, e.g. submit_answer with `[solve(t) for t in test_inputs]`, and keep `answer` as that list."*
  - **Fix A5 (`inputs_prelude`)**: Even on single-test tasks (`n == 1`), `test_inputs = [test_input]` is preloaded in the session because the `run_python` tool description mentions `test_inputs`.
  - **Scaled Budget (`ARC_MULTI_BUDGET=0.5`)**: Scales `max_total_output`, `budget_soft`, and `max_turns` by `scale = 1 + 0.5 * (n - 1)` (e.g., `n=2` gets `120k` hard / `60k` soft / `60` turns; `n=3` gets `160k` hard / `80k` soft / `80` turns).

### 3.2 Zero-Token Identifier Typo Auto-Repair (`auto_repair.py`, Fix I5)
Models generating compact Python exploration code occasionally emit 1–2 character BPE/identifier typos (`scpy` -> `scipy`, `ndiamge` -> `ndimage`, `test_inpt` -> `test_input`, `coms` -> `comps`, `, dbool)` -> `, dtype=bool)`).
- **`pilot_2`**: Returns the raw traceback to the model, spending an entire round-trip generation turn (~1,000–3,500 tokens) to fix a 1-letter typo.
- **Havu `Candidate B32`**: `auto_repair_cell()` runs immediately on any failed `run_python` cell:
  1. Parses `NameError`, `AttributeError` (module or known container/ndarray type), `ModuleNotFoundError`, `ImportError`, and `TypeError` (*unexpected keyword argument*).
  2. Searches `session_names(session) | PRELOADED | _bound_names(tree) | BUILTINS` (or `auto_repair_names.json` + live module introspection) for a **strictly unique** candidate within Optimal String Alignment (OSA) distance $\le 2$ ($\le 1$ for length $\le 4$, `min_len=4` except kwargs, plus `DTYPE_TYPO` positional `dbool`/`dint` $\to$ `dtype=bool`/`dtype=int`).
  3. Runs static AST dataflow analysis (`_Scan` in `auto_repair.py`) over all statements up to the traceback line to prove that **zero** external session objects were mutated (`append`, subscript assignment, `+=`, `heapq.heappush`, `random.shuffle`, `global`, file writes, etc.) before the exception.
  4. Rewrites only `tokenize.NAME` tokens (never strings or comments), re-executes the repaired cell once in the same session, updates `executed[-1]`, and prepends `[auto-corrected: <typo> -> <repl>]\n` to the tool output.

### 3.3 Token-Exact History (`--token-exact`) & Zero Transcript Compaction
- **`pilot_2`**: Re-encodes `st.transcript` from text every turn, stops generation at `"</tool_call>"` (synthetically appending `<|im_end|>`), and once `turn_idx >= 3 and len(st.transcript) > 56000` calls `compact_chatml_transcript`, stripping `<think>...</think>` bodies from all older turns and truncating older `<tool_response>` blocks to 1,500 characters. While this saves prompt characters, it **invalidates the vLLM prefix KV cache** mid-trajectory and alters BPE token splits (~13% of assistant turns tokenize differently when re-encoded from string).
- **Havu `Candidate B32`**: Tracks exact token IDs (`prompt_token_ids + sampled_token_ids`) via `TokensPrompt` with `skip_special_tokens=False` and `spaces_between_special_tokens=False`. Only newly appended `<tool_response>` + `<|im_start|>assistant\n<think>\n` suffix tokens are encoded and concatenated onto the existing token ID list, guaranteeing **100% prefix-cache reuse** and exact token-level continuity. Clean ChatML re-rendering (`resync`) happens only if a turn required `--rescue-code` rewriting or hit `max_tokens` (`finish_reason == "length"`).

### 3.4 Uninterrupted Thinking (`64k` / Turn) vs. 8k 2-Stage Nudge
- **`pilot_2`**: Enforces `turn_think_cap = 8192`. If the model has not emitted a tool call by 8,192 tokens, `pilot_2` injects `XML_THINK_NUDGE_STAGE1` (*"Wait — I have been reasoning for a long time without testing in Python..."*) directly inside `<think>`, gives `2,048` grace tokens, and then forcibly closes `</think>` into `<tool_call><function=run_python><parameter=code>`. On hard ARC-AGI-2 tasks where the SFT-trained model needs 10k–22k tokens of uninterrupted hypothesis derivation on Turn 0 or after falsifying an initial hypothesis, cutting off `<think>` at 8k forces half-baked code experiments.
- **Havu `Candidate B32`**: Sets per-turn `max_tokens = 64000` with `--cap-nudge` **OFF** and `--final-mode session` (no synthetic final generation turn). Also handles spurious mid-`<think>` EOS tokens by continuing up to 4 times (`max_eos_continues=4`).

### 3.5 Budget Pacing (`80k / 40k gentle`), Context-Aware Budget (Fix A9), and `16k` Tool Output
- **`pilot_2`**: `total_token_budget=60,000`, `soft_token_target=20,000`, `"strong"` soft-target warning (*"Past the target: submit your best answer now..."*), and `max_tool_output_chars=3,500`. Printing 3–4 training pair grids or component tables frequently exceeds 3,500 chars and truncates critical diagnostic output.
- **Havu `Candidate B32`**: `max_total_output=80,000` (scaled by `1 + 0.5*(n-1)` on multi-test), `budget_soft=40,000` (scaled), `"gentle"` soft-target wording (*"Past the target: submit as soon as a rule fits every training pair."*), `budget_warn=0.75` (*"Budget nearly spent: submit your best answer now with submit_answer (or at least set `answer`)."*), `turns_note=5`, `max_tool_output=16,000`, `tool_timeout=30s`, and **Fix A9 (`context_budget`)** so when `max_model_len` (`131,072` on Qwen3.8 TPU) is tighter than the remaining output budget (e.g. on 3-input multi-test tasks), budget warnings smoothly adjust to remaining context room and `ContextFull` (`< 1,024` tokens left) triggers clean session fallback instead of an engine error.

### 3.6 Isolated `LocalExecutor` Process Sandbox & Fallbacks (Fixes A1–A11)
- **`pilot_2`**: Executes cells via `exec()` on a `threading.Thread` inside the vLLM Python host process using `sys.settrace` for timeouts (which cannot interrupt C-level NumPy/SciPy operations and can accumulate stuck threads).
- **Havu `Candidate B32`**: Runs one dedicated child Python process (`LocalExecutor`, `style="ipython"`) per active attempt in its own process group (`start_new_session=True`), with `RLIMIT_FSIZE=64MB`, memory cap `4096MB`, `oom_score_adj=1000`, and single-threaded BLAS (`OMP_NUM_THREADS=1`). Timeouts send `SIGINT` to the process group (preserving session variables if caught within 5s grace) and otherwise `SIGKILL` and respawn with `train`, `test_input`, and `test_inputs` cleanly restored.
- **All 11 Audit Fixes (`A1–A11`) Included**:
  - **A1**: Partial multi-test validation (`answer_items` / `solve_items`) so 1 invalid grid in `answer` never discards valid sibling grids, and `solve()` fills only the missing indices.
  - **A2**: `cut_parse` yields text-parsed grids from length-cut turns to session `answer` / `solve()`.
  - **A3**: `solve_items` calls `solve(copy.deepcopy(t))` (Python `list`) first, then automatically retries `solve(np.array(t))` if `solve` expects a NumPy array (`pilot_2`'s E031 fallback only passed a Python `list`).
  - **A4**: Runs session `answer` / `solve()` fallback even if an attempt stops due to an exception/abort.
  - **A5**: Preloads `test_inputs = [test_input]` even on 1-test tasks when `--multi-test` is active.
  - **A6**: `multi_check_code` wraps multi-test submissions into a 1-grid function of `test_input` (`[test_input] + test_inputs[1:]`) so `train_check` labels multi-test programs accurately.
  - **A7**: All host probe cells import `builtins as _b` (`_b.eval`, `_b.compile`, `_b.globals`, `_b.print`) so user cells defining `def eval(...)` or `print = ...` never crash `submit_answer` or fallbacks.
  - **A8**: `restore_inputs` restores host copies of `test_input` and `test_inputs` before fallback harvesting in case user code overwrote `test_input` during training-loop debugging.
  - **A9**: `context_budget` + `ContextFull` guard against exceeding `max_model_len`.
  - **A10**: `_plain` recursively converts nested `np.ndarray` elements (e.g. `[solve(t) for t in test_inputs]` returning list of `ndarray`s) and 0-d `np.bool_` scalars while still rejecting Python `bool`s.
  - **A11**: Chunked `ANSWER_SLICE` transfer when multi-grid JSON exceeds `max_output` (`16,000` chars) + per-grid diagnostic `grid_problem` error messages.

### 3.7 4-Path Zero-Token Post-Attempt `train_check` & `ranked_answers`
- **`pilot_2`**: Only marks `train_pass=True` if `solve(grid)` (or a `submit_answer` expression explicitly referencing `test_input`) passes all training pairs. If the model names its solver `transform(g)`, `apply_rule(g)`, or writes an imperative cell computing `answer` from `test_input`, `pilot_2` labels it unverified (`train_pass=False`).
- **Havu `Candidate B32` (`train_check.py`)**: After the attempt finishes (consuming **0 model tokens**), checks whether the program reproduces the exact SHA-1 hash `ph` of the submitted test prediction on `test_input` and scores it against `train` across 4 cascading paths:
  1. **`expr`** (priority 3): Re-evaluates `submit_code` (wrapped via `multi_check_code` on multi-test) with `test_input` bound to each training input (`10s` per grid).
  2. **`cell`** (priority 2): Finds session variables matching `ph` (`answer` first), locates the last `run_python` cell storing to that variable and loading `test_input`, and re-executes that cell in a shallow copy of session globals per training input (`20s` per grid).
  3. **`fn`** (priority 1): Inspects up to 40 user-defined 1-arg functions in `__main__` globals, verifies reproduction of `ph` on `test_input` first (trying both `list` and `np.array`, `3s` each), and scores matching functions on `train`.
  4. **`transcript`** (priority 0, fallback when earlier paths don't apply): Replays all executed `run_python` / `submit_answer` cells from scratch in a fresh `LocalExecutor` per training input.
- **Ranking (`ranked_answers`)**: Ranks valid candidate grids per test pair by `(train_pass is not True, attempt_idx)` and selects the top-2 distinct grids (falling back to `test_input` if fewer than 2 distinct grids exist).

---

## 4. Experiment Matrix Launched on `final` (`step-00486`)

To give both an **exact replica** of Havu's Candidate B32 (`ARC_ATTEMPTS=2`) and a **strictly controlled `N=4` comparison** against `e031_deepseek_teaching_bo4` (`N=4`), we evaluate both variants on the full 120 ARC-AGI-2 tasks (167 pairs) using `20` parallel `viperfish=2x2x1` shards per variant (`40` WUs total):

1. **`havu_b32` (`N=2` attempts/task)**: Exact Candidate B32 configuration (`ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, `ARC_AUTO_REPAIR=1`, `80k` hard / `40k` soft `"gentle"`, `64k` max_tokens/turn uninterrupted, `--token-exact`, `--rescue-code`, `--lenient-parse`, `--final-mode session`, `--fallback-solve`, `LocalExecutor(style="ipython")`, `GuardedStream` thinking gzip loop guard, 4-path `train_check` + `ranked_answers`). Directly comparable to `e031_deepseek_teaching` (`N=2`: 49.17% Pass@1 / 52.50% Pass@2).
2. **`havu_b32_bo4` (`N=4` attempts/task)**: Identical Candidate B32 harness with `ARC_ATTEMPTS=4` (recording both Havu's exact `ranked_answers` and majority-vote-among-`train_pass` diagnostics). Directly comparable to `e031_deepseek_teaching_bo4` (`N=4`: 56.25% Pass@1 / 61.53% Pass@2).

### 4.1 Launched Experiment

- **XID**: xid/305298492 (`arc_agi2_r2a_final_step486_havu_b32_n2_and_bo4_20shards_v2`, `group:research-mm/genai-lti-lh-tasks`, 40 × `viperfish=2x2x1`, launched 2026-10-28 08:01 UTC).
  - Superseded xid/305292042 (same config): all 40 WUs crashed at their first `enqueue` with `ValueError: JAX does not support per-request seed` (`tpu_inference/platforms/tpu_platform.py:validate_request`). Havu's per-turn `SamplingParams(seed=...)` is GPU-only; the TPU harness now samples with temperature/top_p/top_k only, like `pilot2_eval.py`.
- **Outputs**: `/cns/lu-d/home/arazdai/kaggle/arc_agi_2/r2_results/r2_curated_bs32_lr1e5_xid304951546/final/{havu_b32,havu_b32_bo4}/` (`summary_shard_*.json`, `pairs_*.jsonl`, `attempts_*.jsonl`, `transcripts_*.jsonl`, `submission_*.json`), alongside the earlier `e031_deepseek_teaching{,_bo4}` runs from xid/305119533 on the same checkpoint.
- **Pre-launch verification**: new vLLM-free suite [`havu_b32_harness_test.py`](../pilot_2/havu_b32_harness_test.py) (35 tests: `LocalExecutor` sandbox, `submit_answer` / A1–A11 fallbacks, I5 auto-repair, Qwen XML parse + rescue, A9 budget notes, all 4 `train_check` paths + `ranked_answers`, sharding / resume / queue priority, and scripted end-to-end `_process_completed_turn` rollouts). It caught a launch-killing porting bug: hermetic google3 binaries have `sys.executable is None` on Borg, so Havu's `subprocess.Popen([sys.executable, "-c", ...])` sandbox would have crashed on the first `run_python` call in every work unit. `LocalExecutor` now obtains workers from a pre-TPU zygote fork server ([`havu_sandbox_zygote.py`](../pilot_2/havu_sandbox_zygote.py), forked at the top of `pilot2_eval.py` before `jax` / `vllm` import — the same mechanism already proven on Borg by `pilot_1/.../deepseek_v41_arc2_eval.py`), with identical semantics (own session / process group, `SIGINT` + 5s grace then `SIGKILL` + respawn, `RLIMIT_AS` / `RLIMIT_FSIZE` / `oom_score_adj`, 16k output cap).

---

## 5. Results: Havu Candidate B32 vs. `pilot_2` on R2-A `final` (`step-00486`)

Identical 120 official ARC-AGI-2 evaluation tasks (167 test pairs), identical Block-FP8 checkpoint, official task-weighted Pass@k (per task: mean over its test pairs; summed over tasks). Baselines are xid/305119533; Candidate B32 is xid/305298492 (+ resume xid/305337359 for `bo4` shards 6 and 19). A "try" is one rollout: `pilot_2` runs one per `(task, test pair)` (167×N), B32 one per task with all test inputs preloaded (120×N).

| Harness | N | Official Kaggle Pass@1 (120 tasks) | Official Kaggle Pass@2 (120 tasks) | Pair Pass@1 / Pass@2 (167) | Task-level P@2 W / L / T | Median tokens / try *(solved try)* | Mean / p90 tokens per try | Total gen tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pilot_2` `e031_deepseek_teaching` | 2 | 49.17% (59.00) | 52.50% (63.00) | 49.10% / 51.50% | — | 37,681 *(16,891)* | 36,386 / 60,011 | 12.15M |
| **Havu B32 `havu_b32`** | 2 | **57.36%** (68.83, **+8.19 pp**) | **66.11%** (79.33, **+13.61 pp**) | **55.69% / 65.87%** | **28 / 8 / 84** | **30,410** *(19,466)* | 41,234 / 80,000 | **9.90M** (`~4.4h` on 4×L4; fits the 13.6M Kaggle budget) |
| `pilot_2` `e031_deepseek_teaching_bo4` | 4 | 56.25% (67.50) | 61.53% (73.83) | 52.69% / 59.88% | — | 39,390 *(15,581)* | 36,968 / 60,011 | 24.69M |
| **Havu B32 `havu_b32_bo4`** | 4 | **60.42%** (72.50, **+4.17 pp**) | **72.50%** (87.00, **+10.97 pp**) | **60.48% / 73.05%** | **20 / 5 / 95** | **33,238** *(17,020)* | 39,953 / 80,000 | 19.18M |


Token profile: B32's median try is **~19% shorter** (30.4k vs 37.7k) because 84% of tries end with an accepted `submit_answer` early, while its tail is longer (p90 = the 80k hard cap vs 60k) because the 64k uninterrupted turns are allowed to run to the budget on hard tasks; net total cost is **18% lower at N=2** (9.90M vs 12.15M) and **22% lower at N=4** (19.18M vs 24.69M).

### 5.1 What changed between `pilot_2` (`e031_deepseek_teaching`) and Candidate B32 — short version

1. **One rollout per task instead of per test pair**: all test inputs are shown and preloaded (`test_inputs`), budgets/turns scale ×(1 + 0.5·(n−1)); the model solves 2–4 test inputs with one rule.
2. **No 8k thinking cap / no synthetic `<think>` nudges**: `max_tokens=64k` per turn, uninterrupted; spurious EOS inside `<think>` is continued (≤4×).
3. **Budget 80k hard / 40k soft with "gentle" wording** (was 60k / 20k "strong"), 16k-char tool output (was 3.5k), 30s cell timeout (was 20s), `budget_warn` at 75%, "N turns left" notes.
4. **Token-exact multi-turn history**: exact `prompt_ids + sampled_ids` are fed back (no string re-tokenization, no mid-run transcript compaction) → stable prefix cache, no BPE drift.
5. **`submit_answer` is accepted immediately** (no `verify_submit` re-check that could reject it; no hidden `solve()` checks after each cell).
6. **Isolated per-attempt Python process** (own process group, `RLIMIT_AS` 4 GB, `SIGINT` then `SIGKILL` + respawn) instead of an in-process thread sandbox; on Borg this is served by a pre-TPU zygote fork server.
7. **Zero-token identifier auto-repair** (`NameError`/`AttributeError`/kwarg typos within OSA distance ≤2, AST side-effect check, one retry) + code-outside-tool rescue + name normalisation.
8. **4-path post-attempt `train_check`** (`expr` → `cell` → `fn` → fresh `transcript` replay) and **`ranked_answers`** selection (`train_pass` first, then attempt order; top-2 distinct grids) instead of weighted vote + D8/colour equivariance bonus.
9. **Runtime loop guard** (gzip-ratio monitor on 1k-token windows inside `<think>`) and **smallest-prompt-first scheduling** (all attempt-0s first, then attempt-1 prioritising tasks without a `train_pass`).

### 5.2 Where the gain comes from (`attempts_shard_*.jsonl`, `havu_b32`, 334 pair-attempt rows / 240 rollouts)

- **Clean termination**: 280/334 attempts (83.8%) end with an accepted `submit_answer`; 26 hit the 131k `max_model_len` (`context`, all on multi-test tasks — Fix A9 fallback), 15 `max_turns`, 13 `output_budget`; 34 attempts (10.2%) produced no grid at all (`answer_source=None`).
- **4-path `train_check` recall**: 249/334 attempts (74.6%) labelled `train_pass=True` with 67.1% precision against the hidden test output; label provenance `fn` 138 / `cell` 114 / `transcript` 31 / `expr` 17 — i.e. **~85% of verified labels come from paths `pilot_2` did not have** (it only checked `solve()` / a `test_input` expression). This is what makes `ranked_answers` a strong selector.
- **Rescue machinery is rarely needed on this SFT model**: auto-repair fired 11× (8 succeeded), `--rescue-code` 0×, loop-guard 0×. The accuracy delta is therefore attributable to items 1–5 and 8 above, not to 7 or 9.

### 5.3 Decision for the new SFT round

Adopt Candidate B32 (`havu_b32`, N=2) as the evaluation harness for all future SFT checkpoints and as the Kaggle inference harness: on the same checkpoint it is both more accurate (+13.6 pp Official Pass@2) and cheaper (9.9M vs. 12.15M tokens). Follow-ups worth testing: (a) a 100k hard / 50k soft budget on 1-test tasks (13 `output_budget` stops), (b) prompt-side mitigation for the 26 `context` stops on 3–4-test tasks (e.g. `ARC_MULTI_BUDGET=0.35` or compacting the preloaded `test_inputs` prompt), (c) N=3 as a middle ground (N=4 adds +6.4 pp Pass@2 over N=2 for 1.9× the tokens).

### 5.4 Decomposition of the gain (from `attempts_shard_*.jsonl` / `pairs_shard_*.jsonl`)

| Question | Evidence |
| :--- | :--- |
| Generation or selection? | **Generation.** B32 N=2: Pass@2 66.11% = Oracle(any attempt correct) 66.11% (selection gap 0.0 pp); `pilot_2` N=2: Oracle only 54.44% (gap 1.9 pp). At N=4: B32 Oracle 75.42% vs `pilot_2` 63.47%. |
| Multi-test vs single-test | 45 tasks have 2–4 test inputs. Task-weighted Pass@2, N=2: multi 48.89% → **65.19%** (+16.3 pp), single 54.67% → **66.67%** (+12.0 pp). 17/28 won tasks and 6/8 lost tasks are multi-test; the 3 largest losses burned >100k tokens and died on the 131k context (`context` stop). |
| Budget use | 32% of `pilot_2` attempts ran into the 60k cap (unsolved median 59.6k; 20 turns median under the 8k think cap + nudges). B32 solved rollouts: median 19.8k tokens / 14 turns; only 12% of solves needed >40k and 4% >60k. 84% of B32 attempts end with an accepted `submit_answer`. |
| `train_pass` quality (N=4, 694 rows) | P(correct \| `train_pass=True`) = 70.3%, `False` = 15.7%, `None` = 0% — `ranked_answers` is near-oracle at N=2; the gap opens to 2.9 pp at N=4 (majority-vote-among-verified recovers 0.4 pp). |
| Rescue machinery | Auto-repair 11 fires (8 ok), rescue-code 0, loop guard 0 → not a driver on this SFT model. |
| Run-to-run noise | The N=2 prefix of the N=4 run scores 62.50% vs 66.11% for the dedicated N=2 run → ±3–4 pp at 120 tasks; differences below ~3 pp need repeats. |

### 5.5 Simulated improvements on the N=4 logs (attempts are independent, so prefix truncation is exact)

| Policy | Pass@1 | Pass@2 | Oracle | Tokens | Attempts/task |
| :--- | :---: | :---: | :---: | :---: | :---: |
| fixed N=2 | 54.58% | 62.50% | 62.50% | 9.65M | 2 |
| fixed N=3 | 58.75% | 70.00% | 72.08% | 14.59M | 3 |
| fixed N=4 | 60.42% | 72.50% | 75.42% | 19.18M | 4 |
| **adaptive: 2 attempts, +1 (max 3) unless two verified attempts agree** | 58.75% | **69.17%** | 71.25% | **13.77M** | 2: 44 tasks, 3: 76 |
| adaptive: 2 attempts, +up to 2 (max 4) unless two verified agree | 60.42% | 71.67% | 73.75% | 16.90M | 2: 44, 3: 18, 4: 58 |
| adaptive: extra attempts only while no verified attempt (max 4) | 60.42% | 68.33% | 68.33% | 12.47M | 2: 95, 3: 6, 4: 19 |

Other measured waste: **65/480 rollouts (13.5%) produced no grid at all** (`output_budget` 25, `max_turns` 22, `context` 15) and burned **5.03M tokens (26% of the N=4 total)**; 3 tasks had all four rollouts end with no answer.

### 5.6 Proposed harness modifications (ordered by expected value / risk)

1. **Adaptive N with agreement early-stop** (table above): ≈ +6.7 pp Pass@2 over N=2 for +4.1M tokens, landing at ~13.8M ≈ the Kaggle budget; combine with a lower per-rollout hard cap (64k; only 4–5% of solves need >60k) to get back under 13.6M.
2. **Stop the no-answer bleed**: reserve the last ~4k tokens of every rollout for a forced non-thinking `submit_answer` turn (we had this in `pilot_2`; B32's `--final-mode session` only harvests session variables), inject a hard "submit now" user note at 85% budget, and lower `turns_note` pressure earlier on 4-test tasks.
3. **Multi-test context overflows**: 13 (N=2) / 21 (N=4) rollouts hit 131k context. Options: `ARC_MULTI_BUDGET=0.25` (2-test → 100k), raise `max_model_len` to 163,840 on v6e (KV headroom exists at `max_num_seqs=24`), or compact tool outputs older than 3 turns once the prompt exceeds 100k (not thinking — keep token-exactness for the recent window).
4. **Selector at N≥3**: `train_pass` first, then **vote count among verified candidates**, then `train_frac`, then our D8/colour-equivariance bonus as the final tiebreak (our selector beat `ranked_answers` on single-test Pass@1 at N=4: 64.0% vs 61.3%).
5. **Make the new SFT in-distribution for B32**: render training trajectories in the B32 format (task-level multi-test prompt with `test_inputs`, 16k tool outputs, gentle budget/turn notes, no synthetic nudges, `submit_answer` endings with `train_pass=True`), and up-weight multi-test tasks — the current E031-style data was generated under the `pilot_2` protocol.
6. **Soft verification feedback**: `train_pass=False` submissions are still 15.7% correct, so keep immediate acceptance, but append the per-train-pair failure summary as a tool response and allow one more turn when budget remains (an experiment, not a sure win).

---

## 6. Fidelity audit of our port against Havu's repository (2026-10-29)

Audited against `hav4ik/arcprize2026-arcagi2` @ `main` = `91ecbcc10667d073c011f488c6d54440bf353b01` (2026-10-27, git tag `b32-submission`), using the **hash-pinned** Candidate B32 sources (the live `evals/run_deepseek_tools.py` has moved on by 39 lines: S20 watchdog, E058 image prompts).

### 6.1 Exact file paths that define Candidate B32

| Role | Path in Havu's repo | sha256(8) |
| :--- | :--- | :--- |
| Candidate definition + settings | `havu/kaggle/dsv41_tools/CANDIDATE_B32.md` | — |
| Kaggle solver (scheduler `pick()`, `GuardedStream` loop guard, env knobs, wall-clock deadlines) | `havu/kaggle/dsv41_tools/pinned_candb/solver_dsv41.py` | `77fe80a3` |
| Attempt loop (`solve()`), prompts, A1–A11 helpers, `turn_token_exact`, budget/turn notes, rescue | `havu/kaggle/dsv41_tools/pinned_candb/run_deepseek_tools.py` | `882c817f` |
| 4-path `train_check` + `ranked_answers` | `havu/kaggle/dsv41_tools/pinned_candb/train_check.py` (= `havu/evals/train_check.py` at HEAD) | `8fe36175` |
| I5 auto-repair | `havu/kaggle/dsv41_tools/pinned_candb/auto_repair.py` + `auto_repair_names.json` (= `havu/evals/` at HEAD) | `2292369c`, `571f47eb` |
| Isolated child-process sandbox | `havu/evals/arc_harness/executor.py` (`LocalExecutor`) | — |
| Qwen3.8 rendering (ChatML/XML tools, `tier` system text, EOS-in-thinking continuation, sampling 1.0/0.95/20) | `havu/evals/arc_harness/agent_qwen.py` | — |
| Local 4×L4 replica launch (the run behind "77.4 Kaggle-clock") | `havu/experiments/E055_kaggle_replica/launch_candBf_x2t.sh` | — |
| ARC Prize official prompt template | `havu/notes/prompts/arcprize_official_system_prompt.txt` | — |

B32's effective flags (`solver_dsv41.py` argv + `launch_candBf_x2t.sh` env): `--attempts 2` (**`ARC_ATTEMPTS` default 2**), `--max-turns 40`, `--max-tokens 64000`, `--max-total-output 80000`, `--budget-soft 40000`, `--budget-notes 0.01`, `--budget-warn 0.75`, `--soft-wording gentle` (default), `--turns-note 5`, `--tool-timeout 30`, `--max-tool-output 16000`, `--prompt lean2sub`, `--effort-header minimal`, `--answer-tool`, `--final-mode session`, `--fallback-solve`, `--train-check`, `--token-exact`, `--rescue-code`, `--lenient-parse`, `--multi-test --multi-budget 0.5`, `--context-limit 163840`, `--auto-repair`; loop guard `LOOP_WINDOW/EVERY/RATIO/CONFIRM = 1000/1000/8/10`; `EXEC_MEM_MB 4096`; **off**: `--cap-nudge`, `--verify-submit`, `--best-guess-at/turns` (I7), `--post-pass-note` (I8), `--alt-answer`, `--prior-mode`, `--extra-attempt-if-no-train-pass`. Model-specific: `--model dsv41 --effort low`, `extra_body = {reasoning_effort: 50, temperature: 1.0, top_p: 1.0}` (no per-request seed).

### 6.2 What is byte/AST-identical in our port (`pilot_2/havu_b32_harness.py`, `havu_auto_repair.py`)

- Prompt strings: ARC Prize template, `lean2sub` suffix (`lean2 + _SUBMIT`), `EFFORT_HEADERS["minimal"]`, `TIER_SYSTEM["xhigh"]`, `TOOLS_HEADER/FOOTER`, `GEN_PROMPT`, `MULTI_NOTE`, `RESCUE_NUDGE`, both tool descriptions (`tools_for`).
- AST-identical helpers: `valid_grid`, `grid_problem`, `_items_result`, `answer_items`, `multi_check_code`, `inputs_prelude`, `grid_hash`, `tool_output`; equivalent up to renames / type annotations / `uuid.` prefixes: `submit_answer`, `_answer_value`, `solve_items`, `restore_inputs`, `session_names`, `auto_repair_cell`, `rescue_tool_call`, `rescue_name`, `lenient_parse`, `budget_note`, `turns_note`, `context_budget` (A9), `ranked_answers`, `train_check` (all 4 paths, same timeouts 300 s / 20·(n+1)+30 s / 3 s per fn probe, same `(score, priority, name)` max rule).
- `auto_repair.py`: 25 defs, all constants (`BUILTINS`, `TOP_MODULES`, `KWARGS`, `MUTATORS`, `ARG_MUTATORS`, `ALIASING_CALLS`, `DTYPE_TYPO`, …) identical; `propose()` returns identical rewrites on the regression cases; `auto_repair_names.json` sha256 `571f47eb` matches.
- Attempt loop control flow (`solve()` ↔ `_process_completed_turn` / `_finalize_attempt_session_and_train_check`): per-turn cap `min(64k, remaining, context room)`, rescue-code then ≤3 `RESCUE_NUDGE`s, `cut_parse` (A2), tools skipped on `length`, name rescue, auto-repair on `error`, immediate submit acceptance (multi: list of n grids), budget + turns note appended to the last tool message, fallback order `restore_inputs → answer / answer_items → solve(test_input) → solve_items`, `train_check` via `multi_check_code` for multi-test, `no_answer_0` / `check_error` labels.
- Scheduler `pick()` priorities and the loop guard's thresholds and outcome (abort → budget exhausted → session fallback); token-exact mechanics (`ids + sampled + render(new messages)`, resync on non-canonical/`length`/rescue, `ContextFull` when `limit − len(ids) − 16 < 1024`).

### 6.3 Real differences (and why)

| # | Difference | Impact |
| :-: | :--- | :--- |
| 1 | **Model / format.** B32 is a DeepSeek-V4.1-Flash E128 2-bit configuration (`--model dsv41`, DSML tool syntax, `reasoning_effort 50`, `top_p 1.0`). We apply the B32 *harness* to Qwen3.8-27B using Havu's own Qwen rendering from `agent_qwen.py` (ChatML + `<tool_call>` XML, `tier=xhigh`, temperature 1.0 / top_p 0.95 / top_k 20, EOS-in-thinking continuation ≤4). Havu never ran B32 mechanics on Qwen3.8: his Qwen runs (E031/E032) used `agent_qwen.E031_CONFIG` (60k/20k "strong", 8k think cap) — which is exactly our `pilot_2` baseline. | This is a model transfer of the harness, not a replication of a Havu Qwen number. |
| 2 | **Context limit**: B32 `ARC_CONTEXT_LIMIT=163840` (server `max-model-len 163840`); ours 131072 (TPU vLLM config). | Our 13 (N=2) / 21 (N=4) `context` stops on multi-test tasks are partly an artefact of this; raising to 163840 is the cheapest fix. |
| 3 | **Sandbox transport**: `LocalExecutor` subprocess via `sys.executable` → zygote fork server on Borg; `select()` → `poll()`. Same protocol, limits (`RLIMIT_AS` 4 GB, `RLIMIT_FSIZE` 64 MB, `oom_score_adj`, SIGINT + 5 s then SIGKILL + respawn) and outputs. | Transport only. |
| 4 | **No wall-clock controls / admission gates** (`FORCE_AT_S` 11h10m, `STOP_NEW_S`, `WRAPUP_S`, `ARC_KV_ADMIT_MAX`, 56 workers). Ours: 24 concurrent streams per shard, no deadline. | Offline eval = B32 without Kaggle's clock (Havu measured 77.4 Kaggle-clock vs 79.1 raw on his model). For a Kaggle notebook these must be re-added. |
| 5 | EOS-in-thinking continuation lacks Havu's `low_streak < 2` guard (has a `+256 tokens < budget` guard instead). | 1/240 (N=2) and 6/499 (N=4) rollouts continued at all; negligible. |
| 6 | `rescue_code` additionally accepts a bare Qwen `<parameter=code>` fragment; `train_check` probe templates use `builtins as _b` (A7 shadow-proofing) where Havu's use bare builtins; `ranked_answers` filters `valid_grid(pred)` instead of `pred is not None`. | Strict supersets; 0 rescues fired, no labels differ in practice. |
| 7 | Our run reports additional diagnostics (`mv_pass1/2`, `sel_oracle`, per-pair attempt rows) and the N=4 variant; **B32 itself is N=2** — the N=4 run was only our controlled comparison against our own `bo4`. | None on the N=2 numbers. |
