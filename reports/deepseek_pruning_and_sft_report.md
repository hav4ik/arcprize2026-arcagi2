# DeepSeek-V4.1-Flash-1017 Pruning & SFT Technical Report

**Date:** 2026-10-27  
**Scope:** Comprehensive analysis of collaborator (`havu` / `chankhavu`) experiments (`E042`, `E043`, `E046`, `E051`–`E056`) achieving **72.8%–76.5% Pass@1 / 82.5%–85.6% Pass@2** on ARC-AGI-2 with a pruned + 2-bit quantized `DeepSeek-V4.1-Flash-1017` (`Tool E128 2-Bit`), followed by the exact procedure, code sources, and time estimates for **unquantizing / rebuilding pre-quantized weights** and running **Supervised Fine-Tuning (SFT)**.

---

## Executive Summary & Key Metrics

| Metric / Property | Full Base Model (`Flash-1017`) | Pruned `E72` (4-Bit Native MXFP4) | Pruned `Tool E128` (2-Bit GPTQ) | `Tool E128` + Multi-Test (`E046`) | `Tool E128` + Auto-Repair (`E052`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Routed Experts / Layer** | `384` (top-6) | `72` (top-6) | **`128` (top-6)** | **`128` (top-6)** | **`128` (top-6)** |
| **Expert Precision** | MXFP4 (`4.25 eff. bits`) | MXFP4 (`4.25 eff. bits`) | **`uint2` GPTQ (`g128`/`g64`)** | **`uint2` GPTQ (`g128`/`g64`)** | **`uint2` GPTQ (`g128`/`g64`)** |
| **Engram Host Table** | `188.8 GiB` (100% rows) | `64.0 GiB` (34% rows) | **`64.0 GiB` (34% rows)** | **`64.0 GiB` (34% rows)** | **`64.0 GiB` (34% rows)** |
| **Checkpoint Disk Size** | `510.3 GB` | `118.5 GB` | **`131.1 GB`** | **`131.1 GB`** | **`131.1 GB`** |
| **GPU Weights (TP4 on 4×L4)** | Does not fit (`>120 GiB`) | `16.64 GiB / GPU` | **`16.87 GiB / GPU`** | **`16.87 GiB / GPU`** | **`16.87 GiB / GPU`** |
| **Routing Traffic Covered** | `100.0%` | `69.0%–73.7%` | **`87.5%`** | **`87.5%`** | **`87.5%`** |
| **4×L4 Tool Decode Speed** | N/A | `205 tok/s` (`C=32`) | **`340.5 tok/s` (`C=48`)** | **`318–341 tok/s`** | **`318–341 tok/s`** |
| **Python Identifier Error Rate** | `2.3%` calls | `44.6%` calls | **`14.2%` calls** | **`14.0%` calls** | **`~5.5%` calls (repaired)** |
| **ARC-AGI-2 Eval Pass@1 (120 tasks)** | `74.5%` (API) / `69.5%` (SH) | `29.3%` (tool) / `57.4%` (no-tool) | **`72.8%`** | **`73.7%` task (`74.0%` pair)** | **`76.5%` (vs `71.4%` ctrl)** |
| **ARC-AGI-2 Eval Pass@2 (120 tasks)** | `84.9%` (API) | `66.9%` (no-tool) | **`82.5%`** | **`84.6%` task (`85.6%` pair)** | **`~85%+`** |

---

## Part 1: Collaborator Pruning Pipeline, Components, Hyperparameters & Tricks

### 1.1 Base Architecture & Kaggle 4×L4 Hardware Constraints (`E042` & `E043`)

`DeepSeek-V4.1-Flash-1017` (`revision 1bae0e1088993b9b82de08f59de5be5a34e42170`, `510.31 GB`) is a hybrid MoE + Engram architecture:
* **Backbone (`40` layers, `hidden_size = 5120`)**:
  * **Attention (`9.7 GB` total dense + router + shared experts)**: Multi-head Latent Attention (MLA) interleaved with 7 Sliding-Window Attention (SWA) groups, a sparse FP4 dynamic indexer (`dsv41_indexer.py`), and Multi-head Hyper-Connections (`mHC`). Stored in BF16 / FP8 (`E4M3` with `UE8M0` scales).
  * **MoE FFN (every layer `0..39`)**: `384` routed experts per layer (`top-6` routing with sigmoid gate + bias `bias` / `bias_vl`, `moe_intermediate_size = 2304`, `w1`/`w3` `[2304, 5120]`, `w2` `[5120, 2304]`) stored in native **MXFP4** (`int8` packed 2×E2M1 nibbles + `F8_E8M0` group-32 scale, ~`289 GB` total), plus `1` shared expert per layer in **MXFP8**.
  * **Engram N-Gram Lookup Table (`layers 1` and `14`)**: `384M` rows × `1280` cols in FP8 + scales = **`188.8 GiB`** (`203 GB` on disk), offloaded to pinned host RAM and queried via 24-head prime-modulo n-gram hashes (`2-gram`, `3-gram`, `4-gram`).
  * **Dropped Modules (`--text-only`)**: Vision encoder/aligner (`vision.*`, `aligner.*`, `image_*`) and the 3-layer DSpark speculative drafter (`mtp.0..2`, `7.38 GiB`), saving ~`19 GiB` immediately.

**Why Pruning + Quantization Was Necessary for Kaggle (`4×L4`, `96 GiB` VRAM, `173 GiB` Host RAM cgroup):**
1. The `188.8 GiB` Engram table alone exceeds Kaggle's `173 GiB` host RAM limit.
2. Under Tensor Parallelism `TP=4`, keeping enough KV cache (`>= 1.5M` FP8 KV tokens across `40–48` concurrent streams to avoid prefix-cache thrashing at `>80%` KV utilization) requires model weights to stay under **`~17.0 GiB / GPU`** (`~68 GiB` total across 4 GPUs).

---

### 1.2 Component 1: Lossless Engram Row Frequency Pruning (`188.8 GiB -> 64.0 GiB`)

* **Code**: `havu/experiments/E043_dsv41_real_pruning/scripts/engram_census.py`, `patches/apply_compact_engram.py`, `scripts/export_pruned.py`.
* **Procedure**:
  1. Rebuilt exact token streams (`scripts/tokenize_rollouts.py`) for **10,001 Flash-1017 tool rollouts** (`251.8M` tokens: `106.5M` prompt/tool + `145.3M` model-generated) from `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036`.
  2. Ran an exact numpy port of `inference/engram.py` across all `6.04B` n-gram lookups per layer (`engram_census.py`). Found that only `58.2%` (`223.6M` of `384M` rows = `110 GiB`) are ever touched on ARC traffic, and high-frequency rows dominate (`>64K` hits carry `39%` of lookups).
  3. Selected the **top 34% most frequent rows globally** (`engram_top64gib.json`: `35%` on layer 1, `32%` on layer 14 = **`64.0 GiB`** total, or `15.8 GiB` pinned RAM per TP rank).
  4. Added an `int32` lookup table `layers.{1,14}.engram.embed_tokens.row_map` (`[384M]`, `2.9 GiB` host RAM) mapping kept row IDs to compact indices `0..K-1` and dropped row IDs to `-1` (which `apply_compact_engram.py` reads as a zero vector before $W_{kv}$).
* **Result**:
  * Covers **`97.6%`** of calibration lookups and **`94.0%`** of held-out eval lookups (`~100%` of prompt/grid lookups).
  * **Zero measurable quality loss**: Teacher-forced NLL is `+0.04%` (`0.7133` vs `0.7130` nats/tok, inside run-to-run batching noise `0.0004`), and on 1-attempt ARC-AGI-2 eval, `384 experts + 64 GiB Engram` scored **`69.5%` Pass@1** vs `65.9%` for the full `188.8 GiB` table.

---

### 1.3 Component 2: One-Shot Protected Contribution Expert Pruning (`384 -> 128` Experts/Layer)

* **Code**: `patches/e043_obs.py`, `scripts/replay_reap.py`, `scripts/reap_aggregate.py`, `scripts/special_stats.py`.
* **Why Standard REAP & Iterative Pruning Failed**:
  * **Plain REAP** ranks experts by conditional mean output amplitude when selected ($\mathbb{E}[\|y_e\| \mid e \in \text{top-}6]$). On DeepSeek-V4.1-Flash, this selects rare, high-norm outlier experts that cover only **`32%`** of routing selections at `E64` (`14%` in the worst layer), yielding terrible NLL (`0.922` vs `0.857`).
  * **Iterative pruning** (`384 -> 192 -> 128 -> 96 -> 72 -> 64 -> 48`, re-collecting 1.5M tokens at each stage in `iterative_prune.sh`) performed strictly **worse** than one-shot selection at every budget (`0.879` vs `0.822` at `E72`) because cascading routing shifts compound across stages.
* **The Winning Scoring Rule (`keep_contrib_dec_sp0_w10_e128.npy`)**:
  1. Recorded per-token, per-slot native weighted expert output L2 norms $a_{t,e} = r_{t,e} \| \text{Expert}_e(x_t) \|_2$ on `MarlinExperts.moe_sum` over **10.04M calibration tokens** (`418` rollouts).
  2. Defined the **base contribution score** (`contrib_dec`):
     * **Encoder layers (`0..19`)**: sum of $a_{t,e}$ over **all token positions** (prompt + tool outputs + generated).
     * **Decoder layers (`20..39`)**: sum of $a_{t,e}$ over **model-generated target positions only** (`5.76M` tokens).
  3. **Critical Trick — Tool-Call-Block Upweighting (`w = 10`)**:
     * Tool-call blocks (`</think>` token `128822` through EOS, containing `<｜DSML｜function_calls>...` and Python code) represent only **`~6%`** of generated tokens (`616k` positions in 10M tokens).
     * Under unweighted `contrib_dec` (`w=0`), pruned models drop the specialized experts responsible for closing DSML tags (e.g., emitting a second `<｜DSML｜invoke name="run_p_python">` instead of `</｜DSML｜function_calls>`), causing `0/10` valid tool calls at `E72` and `5/25` length-cut failures even at `E128`.
     * `special_stats.py select --weight 10` scores each expert as:
       $$\text{Score}(L, e) = \text{Contrib}_{\text{dec}}(L, e) + (w - 1) \cdot \text{Contrib}_{\text{tool\_block}}(L, e) \quad (w = 10)$$
     * With `w = 10` (for `E128`; `w = 30` for `E72`), `E128` retains **>95% of tool-block contribution** and achieves **0 DSML format rescues across all 334 ARC-AGI-2 eval attempts**.

* **Why Expert Count Beats Bit-Width (`E128 @ 2-bit` >> `E72 @ 4-bit`)**:
  * DeepSeek-V4.1-Flash has broad routing entropy: **121.7 effective experts/layer** in tool mode (108.4 in no-tool mode). A decode batch of 48 streams touches **~110 distinct experts per layer** on every step.
  * Keeping only `72` experts (`E72` in 4-bit MXFP4) removes ~26–31% of routed traffic and collapses to **`29.3%` Pass@1** in the tool harness.
  * Keeping **`128` experts (`E128`) covers `87.5%` of routing traffic**. When quantized to **2-bit GPTQ**, those ~110 active experts per step require reading **half the HBM bytes** of 4-bit experts, making `E128 2-bit` **both 1.66× faster (`340.5` vs `205 tok/s`) and +43.5 pp more accurate (`72.8%` vs `29.3%` Pass@1)** than `E72 4-bit`!

---

### 1.4 Component 3: Per-Expert 2-Bit Asymmetric GPTQ (`gptq_experts.py` + Humming W2A16)

* **Code**: `scripts/gptq_experts.py`, `scripts/calx_transpose.py`, `scripts/make_lowbit_export.py`, `patches/e043_humming.py`.
* **What Is Quantized vs. Untouched**:
  * **Untouched (exact released precision)**: Attention (`MLA`/`SWA`), router (`ffn.gate.weight`, `bias`, `bias_vl` in BF16/FP32, sliced to 128 rows), shared expert (`MXFP8`), norms, embeddings, LM head, mHC, and the 64 GiB Engram table.
  * **Quantized to 2-Bit (`uint2`)**: Only the `128` kept routed experts per layer (`40 × 128 = 5,120` experts):
    * `w1` (`gate_proj`, `[2304, 5120]`) and `w3` (`up_proj`, `[2304, 5120]`): **`uint2`, group size `g13 = 128`**, `bfloat16` scale `s`, `uint2` zero-point `z` (`DSV41_EXPERT_HUMMING='{"bits":2,"group":128,"zero":true,"scale":"bfloat16"}'`).
    * `w2` (`down_proj`, `[5120, 2304]`): **`uint2`, group size `g2 = 64`**, `bfloat16` scale `s`, `uint2` zero-point `z` (`DSV41_EXPERT_HUMMING_W2='{"bits":2,"group":64,"zero":true,"scale":"bfloat16"}'`).
* **Pruned-Network Calibration & Sequential Layer-Wise Error Feedback**:
  1. Replayed 61 tool rollouts (`1.42M` tokens, sampling 10% of rows = **`142,138` calibration vectors** $X_L \in \mathbb{R}^{142138 \times 5120}$) through the server **with the E128 runtime mask active** (`replay_reap.py --dump-x`), so each layer's MoE inputs reflect the pruned network's actual hidden states.
  2. For each kept expert $e$, formed $H_{13} = X_e^\top X_e \in \mathbb{R}^{5120 \times 5120}$ from tokens actually routed to $e$ and ran column-wise Cholesky GPTQ (`block=128`, MSE shrink search over `{1.0, 0.95, 0.9, 0.85, 0.8, 0.75}`) on $[W_1; W_3]$.
  3. Computed the post-quantization intermediate activations using **DeepSeek-V4.1's exact native clamps and routing weights**:
     $$u = r_e \cdot \left( \text{SiLU}(\min(X_e W_{1,q}^\top, 10.0)) \odot \text{clamp}(X_e W_{3,q}^\top, -10.0, 10.0) \right)$$
     and formed $H_2 = u^\top u \in \mathbb{R}^{2304 \times 2304}$ to GPTQ-quantize $W_2$.
* **Critical Hyperparameter — High Hessian Damping (`percdamp = 1.0`)**:
  * Under top-6 routing across 128 experts, each expert sees a median of only **~3,400 calibration rows** for a `5,120`-dimensional input ($N < d$, rank-deficient Hessian).
  * With standard GPTQ damping (`percdamp = 0.01`), Cholesky inversion severely overfits the in-sample nullspace: on held-out tokens, `percdamp=0.01` had **worse** output error than Round-To-Nearest (`15.0%` vs `13.2%` on Layer 30 `w2`).
  * Setting **`--percdamp 1.0`** (adding $1.0 \times \text{mean}(\text{diag}(H))$ to the diagonal) regularizes the inverse Hessian and slashes relative output error $\text{tr}(\Delta W H \Delta W^\top) / \text{tr}(W H W^\top)$ across all 40 layers to:
    * **`w13`: `2.5%`** (vs `10.9%` RTN — a **4.4× error reduction**)
    * **`w2`: `5.5%`** (vs `12.7%` RTN — a **2.3× error reduction**)

---

### 1.5 Root Cause of Pruned Model Mistakes: BPE Sub-Word Split-Mixing Typos (`analysis/typos/REPORT.md`)

A deep offline audit (`havu/experiments/E043_dsv41_real_pruning/analysis/typos/REPORT.md`) of 29,136 `run_python` calls across full and pruned models revealed a critical finding that directly informs **how to design SFT for the pruned model**:

1. **Pruning (Not 2-Bit Quantization) Causes Surface-Form Identifier Typos**:
   * Logic errors (`ValueError`, `IndexError`, `KeyError`) in `E128` remain at the full-model baseline (`1.69%` vs `1.06%–1.40%`).
   * However, **identifier errors** (`NameError`, `AttributeError`, `ModuleNotFoundError`, bad kwargs) jump **35×** from `0.23%` (full model) to **`8.22%`** (`E128`), `18.70%` (`E112`), `22.84%` (`E96`), and `31.26%` (`E72` in 4-bit MXFP4 without GPTQ!).
2. **Exact Token-Boundary Mechanism ("Split-Mixing")**:
   * **94%** of misspelled identifiers in `E128` are multi-token words in context (vs 18% of all identifiers).
   * In **92%** of `E128` typos, the **first token is 100% correct**, and in **88%**, the divergence begins **strictly at a word-internal continuation (`WIC-alnum`) token boundary** (specifically at token index 1 in 82% of cases).
   * **Why?** The V4.1 BPE tokenizer splits the same identifier differently depending on the preceding character:
     * After a space: ` comps` $\to$ `['Ġcomp', 's']` (**0 typos in 5,971 occurrences!**)
     * After `=`, `(`, or `,`: `=comps` $\to$ `['=', 'com', 'ps']` (**15% typo rate in E128, 81% in E96!**)
   * When re-tokenizing a variable name after punctuation, the pruned model emits the first piece of the punctuation split (`com`) followed by the continuation token of the space-prefixed split (`s`), producing `coms` (`648` times), `scpy` (`sc` + `py` instead of `Ġsc` + `ipy`), `dequeue` (`de` + `queue`), `zzip` (`(z` + `zip`), `mmask` (`=m` + `mask`), or `dbool` (`,d` + `bool` dropping `type=`).
   * Furthermore, the model immediately notices its own slip during generation and writes inline self-repairs like `mash if False else mask` (`7.18%` of calls in `E128`) or `from scpy? no`, which in turn trigger `SyntaxError`s!

---

### 1.6 Inference, Harness & Serving Tricks That Push Pass@1 from `72.8%` to `76.5%+`

1. **`sub80s40` Budget Preset + Minimal Effort Header (`E030` / `E043`)**:
   * `max_total_output_tokens = 80,000`, `soft_budget = 40,000` (injects wrap-up prompt after 40k tokens), `max_tokens_per_turn = 64,000`, `max_turns = 40`, `temperature = 1.0`, `top_p = 1.0`, `reasoning_effort = 50` (low) + minimal system header.
2. **Token-Exact Multi-Turn Continuation (`--token-exact`, `E046`)**:
   * About 1 in 8 completions is sampled non-canonically (re-tokenizing the text string shifts token boundaries by 1–2 tokens, invalidating the vLLM prefix cache and increasing split-mixing risk). Passing raw `prompt_token_ids` accumulated from exact sampled `token_ids` keeps prefix-cache hit rates at **`96.7%`**.
3. **Multi-Test Single-Rollout (`--multi-test --multi-budget 0.5 --fallback-solve`, `E046` Run 7)**:
   * Instead of running separate rollouts for each test input of a multi-test task (`39` of the `120` ARC-AGI-2 tasks have 2–4 test inputs), presents all test inputs in one shared conversation with `+50%` extra token budget per additional test pair.
   * **Result**: Lifts `E128 2-bit` on 120 eval tasks from **`72.8% -> 73.7%` Task Pass@1** (`74.0%` pair Pass@1) and **`82.5% -> 84.6%` Task Pass@2** (`85.6%` pair Pass@2) while **reducing mean tokens per task-attempt by `-17%` overall (`49.0k -> 40.8k`) and `-30%` on multi-test tasks (`77.8k -> 54.8k`)**.
4. **Runtime AST/Fuzzy Identifier Auto-Repair (`evals/auto_repair.py`, `E052`)**:
   * On `NameError`, `AttributeError`, `ImportError`, `ModuleNotFoundError`, or unexpected-keyword `TypeError`, parses the traceback and AST, resolves the failing identifier to a unique OSA edit-distance $\le 2$ match in scope/builtins/module `dir()`, re-executes once (up to 3 chained repairs), and prepends `[auto-corrected: coms -> comps]` to the tool output so the model doesn't repeat the typo in future turns.
   * **Result**: Resolves **61% of reproduced identifier errors on the spot** (`89%–93%` identical to the model's next-turn fix), saving `0.75` turns/attempt on `E128` and boosting **Task Pass@1 to `76.5%`** (vs `71.4%` control on 4×RTX 4090).
5. **Candidate B Serving Stack (`E051` KV CPU Offload + `E054` Indexer Skip + `E056` Kit)**:
   * **E051 32 GiB Host KV Offload**: Offloads evicted KV blocks to 32 GiB pinned host RAM (`swa_store_policy="checkpoint"`, 128-token aligned), achieving byte-identical KV parity up to 120k context and **3.4×–4.0× higher generation throughput** at 56–72 concurrent streams when GPU KV cache fills up.
   * **E054 Grid-Stride Indexer Context Skip (`dsv41_indexer.py`)**: Replaces the decode sparse-attention indexer kernel with a grid-stride context skip (`DSV41_INDEXER_SKIP=1`), cutting decode step latency by **`-5.7%` to `-6.5%`** with **bit-identical logits**.

---

## Part 2: How to Unquantize / Rebuild Pruned DeepSeek & Run SFT (Procedure, Code & Time Estimates)

### 2.1 Crucial Finding: You Have Two Ways to Get Unquantized (BF16) Pruned E128 Weights

In your prompt you noted: *"currently we only have weights saved in 2 bit quantized so we need to unquantize them"*.  
Checking Vu's latest docs (`havu/experiments/E043_dsv41_real_pruning/PREQUANT_E128.md` and `scripts/verify_prequant.py`, committed `2026-10-26`) reveals an important distinction:

1. **What `DeepSeek-V4.1-Flash-1017` was originally released in**:
   * DeepSeek released the base model with **routed experts in MXFP4** (4-bit E2M1 float nibbles + `F8_E8M0` power-of-two block-32 scales) and **dense attention / shared experts in FP8 (`E4M3` + `F8_E8M0` scales) / BF16**.
   * **Every MXFP4 and MXFP8 weight decodes into `bfloat16` with ZERO rounding error (mathematically bit-exact)** because every E2M1 / E4M3 mantissa fits inside BF16's 7-bit mantissa and `UE8M0` has the exact same 8-bit exponent range as `bfloat16`!
2. **What the `Tool E128 2-bit` HF/Kaggle repo (`131.1 GB`) contains**:
   * Every non-routed-expert tensor (attention, router sliced to 128 experts, shared expert, norms, embeddings, LM head, 64 GiB compact Engram) is **already stored in its original, unquantized base precision (BF16 / FP8), byte-for-byte identical to the base model**.
   * Only the `40 × 128` routed experts were replaced by 2-bit Humming tensors in `gptq/layer_00..39.safetensors` (`e{0..127}.{w1,w2,w3}.{q,s,z}`).
   * Because 2-bit quantization is lossy (`cosine ≈ 0.89–0.93` vs the original MXFP4 experts, `~42%` relative weight RMSE), **you have two paths to get a BF16 pruned E128 checkpoint for SFT**:
     * **Path A (Recommended — Lossless Pre-2-Bit Rebuild + Exact BF16 Decode)**: Rebuild the exact pre-2-bit E128 checkpoint on CPU by slicing the kept 128 experts per layer from the base `DeepSeek-V4.1-Flash-1017` checkpoint (which we already have locally in Google3 at `/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1` or on HF `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10`) using the 40 KB manifest (`e043_manifest.json`) from the 2-bit repo, then decode MXFP4 $\to$ BF16 exactly. **Zero 2-bit quantization noise!**
     * **Path B (Direct Dequantization of the 2-Bit `gptq/` Weights to BF16)**: Unpack the 2-bit `uint2` (`q, s, z`) tensors from `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` directly into `bfloat16` via $W = (q - z) \times s$. Useful if you only have the 131 GB 2-bit checkpoint on disk or want SFT to specifically adapt to the 2-bit quantized expert weights.

Below are the exact procedures and code for **both Path A and Path B**.

---

### 2.2 Path A (Recommended): Lossless Rebuild of Pre-Quantized `Tool E128` + Exact BF16 Decode

#### Step A1: Rebuild the Exact Pre-Quantized E128 Checkpoint on CPU (`178 GB` MXFP4/FP8)
* **Source scripts**: `havu/experiments/E043_dsv41_real_pruning/PREQUANT_E128.md`, `scripts/verify_prequant.py`, `scripts/export_pruned.py`.
* **Hardware needed**: CPU only, ~`16 GB` RAM, ~`690–760 GB` disk (or zero base download if reading `/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1`).
* **Commands**:
```bash
export HF_TOKEN=<token_with_read_access_to_chankhavu>
E043=/path/to/arcprize2026-arcagi2/havu/experiments/E043_dsv41_real_pruning

# 1. Download base (or point BASE_DIR to /tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1)
#    and download ONLY the 64 GiB Engram + JSON manifests (~72 GB) from the 2-bit repo:
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir base
hf download chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit \
    --include "model-engram-*" "*.json" --local-dir e128_2bit

# 2. Extract the [40, 384] boolean keep mask (keep_contrib_dec_sp0_w10_e128.npy)
#    and verify its SHA256 (f4c946c3...a56d) against config.json:
python $E043/scripts/verify_prequant.py keep \
    --manifest e128_2bit/e043_manifest.json \
    --config e128_2bit/config.json \
    --out keep_tool_e128.npy

# 3. Export the pre-quantized E128 model (renumbers kept experts 0..127, slices router
#    ffn.gate.{weight,bias,bias_vl} to 128 rows, drops DSpark/vision, links 64 GiB Engram):
python $E043/scripts/export_pruned.py \
    --src base \
    --expert-keep keep_tool_e128.npy \
    --engram-from e128_2bit \
    --text-only \
    --out e128_prequant

# 4. Verify byte-for-byte parity of all 5,120 experts + router rows against base:
python $E043/scripts/verify_prequant.py check \
    --export e128_prequant --base base --all \
    --gptq hf:chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit@main
```
* **Output of Step A1**: `e128_prequant/` (`178 GB` total: `9.7 GB` backbone + `96.3 GB` MXFP4 routed experts + `71.8 GB` compact Engram).

#### Step A2: Exact Decode of Pre-Quantized `MXFP4` Experts & `MXFP8` Dense Tensors to `BF16`
* **Source scripts**: `patches/e043_humming.py` (`mxfp4_to_bf16`) and `google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm/ds4_ref.py` (`dequant_mxfp4`, `dequant_mxfp8`).
* **Exact PyTorch Decoder (GPU or CPU)**:
```python
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def mxfp4_to_bf16(weight: torch.Tensor, scale: torch.Tensor, device="cuda") -> torch.Tensor:
    """
    Bit-exact decode of DeepSeek-V4.1 MXFP4 expert weights to BF16.
    weight: int8/uint8 [N, K // 2] (low nibble = even col, high nibble = odd col)
    scale:  uint8/F8_E8M0 [N, K // 32] (exponent e -> 2^(e - 127) per 32 inputs)
    Returns: torch.bfloat16 [N, K]
      - w1, w3: [2304, 5120] in BF16 (23.59 MB per matrix)
      - w2:     [5120, 2304] in BF16 (23.59 MB per matrix)
    """
    w = weight.to(device).view(torch.uint8)
    codes = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(*w.shape[:-1], w.shape[-1] * 2).long()
    mag = E2M1.to(device)[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    s = scale.to(device).view(torch.uint8).to(torch.int32)
    return (val * torch.exp2((s - 127).float()).repeat_interleave(32, dim=-1)).to(torch.bfloat16)

def mxfp8_to_bf16(weight: torch.Tensor, scale: torch.Tensor, device="cuda") -> torch.Tensor:
    """
    Bit-exact decode of DeepSeek-V4.1 FP8 (E4M3fn + UE8M0 block-32 scale) dense/shared-expert weights to BF16.
    weight: float8_e4m3fn [N, K], scale: uint8/F8_E8M0 [N, K // 32]
    """
    w = weight.to(device).view(torch.float8_e4m3fn).to(torch.float32)
    s = scale.to(device).view(torch.uint8).to(torch.int32)
    return (w * torch.exp2((s - 127).float()).repeat_interleave(32, dim=-1)).to(torch.bfloat16)
```
* **Size in BF16**:
  * All `40 × 128 = 5,120` routed experts in BF16: $5120 \times 3 \times 2304 \times 5120 \times 2\text{ B} = \mathbf{362.39\text{ GB}}$ (`337.5 GiB`).
  * Dense backbone + shared experts in BF16: **`~18.5 GB`**.
  * Compact Engram (kept frozen on CPU or BF16 buffer): **`64.0 GiB`**.

---

### 2.3 Path B: Direct Unquantization (Dequantizing the 2-Bit `uint2` GPTQ Weights to BF16)

If you only have the 2-bit checkpoint (`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`, `131.1 GB`) and want to directly unpack `gptq/layer_00..39.safetensors` back into standard `bfloat16` `nn.Linear` / MoE weights `layers.{L}.ffn.experts.{0..127}.{w1,w2,w3}.weight`:

* **How the 2-Bit Humming Tensors Are Stored (`scripts/gptq_experts.py` & `scripts/verify_prequant.py`)**:
  * Each `gptq/layer_{L:02d}.safetensors` holds `9 × 128 = 1,152` tensors: `e{0..127}.{w1,w2,w3}.{q,s,z}`.
  * `q`: `int32` `[N, K // 16]` — 16 unsigned 2-bit codes (`0..3`) packed LSB-first per `int32` along input dim $K$.
  * `s`: `bfloat16` `[N, K // group]` — per-group scale (`group = 128` for `w1`/`w3`, `group = 64` for `w2`).
  * `z`: `int32` `[N // 16, K // group]` — 16 unsigned 2-bit zero-points packed LSB-first **along output dim $N$** (transposed before packing in `gptq_experts.py:pack()`).
  * Dequantization formula: $W_{\text{bf16}} = \text{bfloat16}\big((q - z) \times s\big)$.

* **Standalone PyTorch 2-Bit Unpacker (No `humming` Kernel Required — Verified in `verify_prequant.py:gptq_to_f32`)**:
```python
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHIFTS = torch.arange(0, 32, 2, dtype=torch.int32)  # 16 x uint2 per int32, LSB first

def unpack_uint2(packed_int32: torch.Tensor) -> torch.Tensor:
    """Unpacks int32 [..., M] -> uint8 [..., M * 16] with values in {0, 1, 2, 3}."""
    shifts = SHIFTS.to(packed_int32.device)
    return ((packed_int32.unsqueeze(-1) >> shifts) & 3).reshape(*packed_int32.shape[:-1], -1)

def dequant_gptq_expert_tensor(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, group: int) -> torch.Tensor:
    """
    q: int32 [N, K // 16]
    s: bfloat16 [N, K // group]
    z: int32 [N // 16, K // group] (packed along N!)
    Returns: exact dequantized bfloat16 weight [N, K]
    """
    q_u2 = unpack_uint2(q).to(torch.float32)                           # [N, K]
    z_u2 = unpack_uint2(z.T.contiguous()).T.contiguous().to(torch.float32)  # [N, K // group]
    s_f32 = s.to(torch.float32)                                        # [N, K // group]
    w_f32 = (q_u2 - z_u2.repeat_interleave(group, dim=-1)) * s_f32.repeat_interleave(group, dim=-1)
    return w_f32.to(torch.bfloat16)

def unquantize_layer_experts(gptq_dir: str, layer_idx: int, n_experts: int = 128, device: str = "cuda"):
    """Unquantizes all 128 experts of layer_idx from 2-bit Humming format to BF16."""
    path = Path(gptq_dir) / f"layer_{layer_idx:02d}.safetensors"
    bf16_experts = {}
    with safe_open(str(path), framework="pt", device=device) as f:
        for e in range(n_experts):
            for proj, group in (("w1", 128), ("w3", 128), ("w2", 64)):
                q = f.get_tensor(f"e{e}.{proj}.q")
                s = f.get_tensor(f"e{e}.{proj}.s")
                z = f.get_tensor(f"e{e}.{proj}.z")
                w_bf16 = dequant_gptq_expert_tensor(q, s, z, group)
                bf16_experts[f"layers.{layer_idx}.ffn.experts.{e}.{proj}.weight"] = w_bf16.cpu()
    return bf16_experts
```
*(Alternatively, if running inside the box venv with `humming` installed, `humming.utils.weight.dequantize_weight(q, s, z, None, humming.dtypes.uint2, packed=True)` performs the exact same operation on GPU, verified to `max_abs == 0.0` in `gptq_experts.py:247`).*

---

### 2.4 How to Run SFT on the Pruned DeepSeek Model (`Tool E128`)

Once the pruned E128 model is unpacked/decoded to BF16 (via **Path A** or **Path B**), there are **two concrete SFT strategies**, depending on whether you want **fast healing of pruning typos** or **full trajectory SFT**:

#### Strategy 1 (Fastest, Lowest Risk & Highest ROI): "Healing LoRA" on Non-Expert Backbone + Shared Experts (With `WIC-alnum` Loss Weighting)
* **Why This Is Ideal First**:
  1. In `Tool E128 2-bit`, **every dense attention layer (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), router (`ffn.gate`), and shared expert (`shared_experts.w1/w2/w3`) is NOT 2-bit quantized**—they are served directly in BF16 / FP8!
  2. If you train LoRA adapters (`rank = 32` or `64`, `lora_alpha = 64`) on **attention + shared experts** (keeping the 128 routed experts frozen in BF16 or even via custom 2-bit forward pass), **you can merge the trained LoRA directly into the dense/shared-expert shards of `DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` WITHOUT even re-running 2-bit GPTQ on the routed experts!**
  3. Even better, per `analysis/typos/REPORT.md` (§8–9, Proposal #5), upweighting the cross-entropy loss by **$\times 10\text{--}20$ on word-internal alphanumeric continuation (`WIC-alnum`) tokens** (the ~6% of tokens where a word is split across BPE tokens, e.g., `com|ps`, `sc|ipy`) directly trains the pruned network to stop making BPE split-mixing typos.

* **LoRA SFT Hyperparameters (Adapted from `google3/experimental/users/arazdai/kaggle/sft/train_sft.py`)**:
  * **Hardware**: `8× NVIDIA B200` (`1,440 GB` HBM3e total — holds the entire `380 GB` BF16 E128 model across 8 GPUs under FSDP2 at `~48 GB/GPU` for weights, leaving `~130 GB/GPU` for long-context activations!).
  * **Target Modules**: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, `shared_experts.w1`, `shared_experts.w2`, `shared_experts.w3` (and optionally routed `experts.*.w1/w2/w3` if followed by GPTQ re-quantization).
  * **Rank / Alpha**: `r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`.
  * **Learning Rate & Schedule**: `lr = 1e-4` (LoRA) with cosine decay, `warmup_ratio = 0.05`, `weight_decay = 0.01`, `bf16 = True`.
  * **Context Length**: `max_seq_length = 65,536` (with sequence/flash attention & activation checkpointing).
  * **Loss Masking**: Assistant-only completion loss (`response_only_loss = True`), with **$\times 10$ token-level loss multiplier on `WIC-alnum` tokens and DSML tool-call tags**.
  * **Dataset**: `1,500–5,000` curated, zero-typo, verified-correct multi-turn `DeepSeek-V4.1-Flash-1017` tool rollouts from `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (or our `deepseek_v41_top10k_rollouts` + `arc_sft_v1_clean.jsonl` in `ARC_AGI2_SFT_DATA_CATALOG.md`), formatted with the exact `encoding/encoding.py` DSML tool chat template (`effort=50`).

#### Strategy 2: Full / MoE-Expert SFT on 8×B200 + Re-Quantization to 2-Bit GPTQ
If you update the **128 routed experts per layer** (either via full SFT in `train_full_sft.py` or LoRA merged into the routed experts):
1. **Unpack to BF16 (Path A recommended)**: Decode `e128_prequant` MXFP4 experts to BF16 (`362.4 GB` experts + `18.5 GB` backbone = `~381 GB` parameters in BF16).
2. **Shard across 8×B200 with PyTorch FSDP2 (`train_full_sft.py`)**:
   * Full BF16 weights + FP32 master weights + AdamW 8-bit (`bitsandbytes` paged AdamW8bit or FSDP2 CPU-offloaded optimizer states):
     * Model weights in BF16: `47.6 GB / GPU` across 8 B200s.
     * Gradients in BF16: `47.6 GB / GPU`.
     * Optimizer states (8-bit AdamW or CPU-offloaded FP32): `~24 GB / GPU` (or `0 GB` GPU with FSDP `CPUOffloadPolicy`).
     * Total static GPU memory on 8×B200 (`180 GB/GPU`): **`~119 GB / GPU`**, leaving `~60 GB / GPU` for activation checkpointing at `32k–64k` context!
   * Hyperparameters for Full SFT: `lr = 5e-6` to `1e-5`, `epochs = 1–2`, `global_batch_size = 16–32`, `max_grad_norm = 1.0`, `freeze_engram = True` (keep the 64 GiB Engram table frozen).
3. **Re-Quantize Updated Routed Experts Back to 2-Bit Humming Format (Mandatory for 4×L4 Deployment)**:
   * Because Kaggle 4×L4 only fits 2-bit routed experts (`16.87 GiB/GPU`), any SFT that modifies the routed expert weights `layers.L.ffn.experts.0..127.{w1,w2,w3}` must be re-quantized to `uint2` (`g128`/`g64`) using `scripts/gptq_experts.py`:
   ```bash
   # Step 1: Dump calibration inputs X_L (or reuse the 142k-row calx_by_layer dump if drift is small,
   #         or re-dump 61 rollouts in ~15 min with replay_reap.py --dump-x)
   # Step 2: Run per-expert GPTQ with percdamp=1.0 in parallel across 8 GPUs (5 layers per GPU):
   for g in 0 1 2 3 4 5 6 7; do
     CUDA_VISIBLE_DEVICES=$g python $E043/scripts/gptq_experts.py \
       --calx /dev/shm/e043/calx_by_layer \
       --model-dir /path/to/sft_e128_checkpoint \
       --layers $((g*5)) $((g*5+1)) $((g*5+2)) $((g*5+3)) $((g*5+4)) \
       --bits13 2 --bits2 2 --group13 128 --group2 64 \
       --percdamp 1.0 \
       --out /dev/shm/e043/gptq_sft_e128_u2 &
   done
   wait

   # Step 3: Package into the 131 GB Humming GPTQ-only deployment layout:
   python $E043/scripts/make_lowbit_export.py \
     --base /path/to/sft_e128_checkpoint \
     --precomputed /dev/shm/e043/gptq_sft_e128_u2 \
     --precomputed-ids compact \
     --keep keep_tool_e128.npy \
     --schema '{"bits":2,"group":128,"zero":true,"scale":"bfloat16"}' \
     --schema-w2 '{"bits":2,"group":64,"zero":true,"scale":"bfloat16"}' \
     --name E128-SFT \
     --out export/tool_e128_sft_gptq
   ```

---

### 2.5 Detailed Step-by-Step Time & Resource Estimates

| Stage | Operation | Hardware Required | Time Estimate | Disk / Memory Footprint |
| :--- | :--- | :--- | :---: | :--- |
| **1a. Pre-Quant Rebuild (Path A)** | Download/link base + 2-bit manifest & run `export_pruned.py` | CPU (`16 GB` RAM, NVMe / `/dev/shm`) | **`15–25 min`** (`~5 min` export if base is local) | `178 GB` output (`96.3 GB` MXFP4 experts + `71.8 GB` Engram + `9.7 GB` backbone) |
| **1b. Or 2-Bit Unpack (Path B)** | Run `unquantize_layer_experts()` on `gptq/layer_00..39.safetensors` | 1 GPU or 16-core CPU | **`4–8 min`** (GPU) / `15 min` (CPU) | Reads `23.6 GB` of 2-bit experts $\to$ writes `362.4 GB` BF16 experts |
| **2. MXFP4/FP8 $\to$ BF16 Decode** | Decode `e128_prequant` to full BF16 safetensors (or decode on-the-fly in loader) | 8 GPUs in parallel (or on-load) | **`5–10 min`** (disk write) / **`0 min`** (on-the-fly in loader) | `381 GB` BF16 backbone+experts + `64 GiB` Engram |
| **3. SFT Data Preparation** | Filter zero-typo winning tool traces (`1.5k–3k` trajectories) + `tokenize_rollouts.py` + `WIC-alnum` mask | CPU (32+ cores) | **`10–15 min`** | `~2–4 GB` tokenized Parquet / Arrow |
| **4a. Option 1: Healing LoRA SFT** | LoRA (`r=32`) on Attention + Shared Experts (`1.5k` traces, `~40M` tokens, 1 epoch) | `8× NVIDIA B200` (FSDP2) | **`45–90 min`** | `~55 GB / GPU` HBM; **No GPTQ re-quantization needed** if routed experts stay frozen! |
| **4b. Option 2: Full / Routed-Expert SFT** | Full BF16 SFT or Expert-LoRA (`1.5k–3k` traces, `40M–80M` tokens, 1 epoch) | `8× NVIDIA B200` (FSDP2 + 8-bit AdamW / CPU offload) | **`2.0–3.5 hours`** | `~120–145 GB / GPU` HBM |
| **5. Post-SFT 2-Bit GPTQ Re-Quantization** *(Only if routed experts were modified)* | Calibration dump (`replay_reap.py --dump-x` on 61 rollouts) + `gptq_experts.py --percdamp 1.0` + `make_lowbit_export.py` | `8× B200` or `8× RTX PRO 6000` | **`25–35 min`** (`~15 min` calx dump + `~6–10 min` GPTQ on 8 GPUs + `5 min` export) | `131.1 GB` final 2-bit checkpoint |
| **6. Verification & Dev Screen** | `check_lowbit_export.py` + 64-trace Teacher-Forced NLL (`nll_eval.py`) + 32-task Dev ARC Probe | `4×L4` or `4×4090` / `B200` | **`20–30 min`** | Verifies NLL (`< 0.796`), valid DSML calls (`25/25`), and typo rate reduction (`< 8.2%`) |
| **Total End-to-End Wall Time** | **Path A + Healing LoRA (Dense/Shared)** vs. **Path A + Full Expert SFT + GPTQ** | `8× B200` | **`~1.5–2.5 hours`** (LoRA) / **`~3.5–5.0 hours`** (Full Expert SFT) | Outputs drop-in `131.1 GB` model for Kaggle `4×L4` |

---

## Reference File Index

* **Collaborator Pruning & Low-Bit Scripts (`hav4ik/arcprize2026-arcagi2`)**:
  * `havu/experiments/E043_dsv41_real_pruning/README.md` — Complete `E043` experimental log (`E128 2-bit`, `E96 2/3-bit`, `E72`, Engram census, REAP vs contribution scoring).
  * `havu/experiments/E043_dsv41_real_pruning/PREQUANT_E128.md` — CPU rebuild guide for the exact pre-quantization `Tool E128` checkpoint.
  * `havu/experiments/E043_dsv41_real_pruning/scripts/verify_prequant.py` — Manifest extractor, byte-parity verifier, and standalone `mxfp4_to_f32` / `gptq_to_f32` decoders.
  * `havu/experiments/E043_dsv41_real_pruning/scripts/export_pruned.py` — Physical pruned checkpoint exporter (renumbers experts `0..127`, slices router, builds `row_map` for compact Engram).
  * `havu/experiments/E043_dsv41_real_pruning/scripts/gptq_experts.py` — Per-expert asymmetric `uint2` GPTQ quantizer (`--percdamp 1.0`, native clamps, Humming packing).
  * `havu/experiments/E043_dsv41_real_pruning/scripts/make_lowbit_export.py` — Packages `gptq/layer_LL.safetensors` into the 131.1 GB deployment checkpoint.
  * `havu/experiments/E043_dsv41_real_pruning/patches/e043_humming.py` — vLLM Humming integration & exact GPU `mxfp4_to_bf16` loader wrapper.
  * `havu/experiments/E043_dsv41_real_pruning/analysis/typos/REPORT.md` — Token-level BPE split-mixing typo analysis & `WIC-alnum` weighting / healing LoRA specification.
  * `havu/experiments/E046_multi_test_rollout/README.md` — Multi-test single-rollout harness (`73.7%` Task Pass@1 / `84.6%` Task Pass@2).
  * `havu/evals/auto_repair.py` & `havu/experiments/E056_candidate_b/README.md` — Runtime AST/OSA typo auto-repair (`76.5%` Pass@1) and Candidate B (`E051` KV CPU offload + `E054` indexer context skip).
* **Google3 Workspace SFT & Loader Files**:
  * `//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm/ds4_load.py` & `ds4_ref.py` — Local checkpoint loader (`/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1`) and `dequant_mxfp4` / `dequant_mxfp8` reference implementations.
  * `//experimental/users/arazdai/kaggle/sft/train_full_sft.py` & `train_sft.py` — 8×B200 FSDP2 full SFT trainer and LoRA SFT trainer.
  * `//experimental/users/arazdai/kaggle/arc_agi_2/reports/ARC_AGI2_SFT_DATA_CATALOG.md` — SFT dataset catalog (`arc_sft_v1_clean.jsonl`, `deepseek_v41_top10k_rollouts`).
