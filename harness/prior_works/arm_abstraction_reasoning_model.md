# ARM: Accelerated (formerly "Abstraction") Recursive Reasoning Model

- **Author:** Marcel Siegert (GitHub `MarcelxMensch`). Leaderboard date 23 Jun 2026.
- **Code:** https://github.com/MarcelxMensch/arm-arc-agi, HEAD `2228242`. The repo has 3 commits, made 23–28 Jun 2026.
  - `06ada3c` renamed the model from "Abstraction Reasoning Model" to "Accelerated Recursive Reasoning Model".
  - `2228242`: "reframe README around acceleration not abstraction".
- **Paper/blog:** none. The leaderboard's `paperUrl` is null, and a web search found nothing.
- **License:** **none**: no LICENSE file, and GitHub reports none. The code is derived from the TRM/HRM codebases.

Source tags: [R] = README, [C] = code/configs, [E] = shipped eval JSONs and `submission.json`, [O] = our analysis.

## TL;DR
- **What it is:** a **trained ~10.8M-param recursive transformer**, a variant of TRM (Tiny Recursive Model). It is **not an LLM harness**: there are no prompts and no API calls. [R,C]
  - Each puzzle gets a learned embedding; answers come from voting over ~1,000 augmentations.
- **Change vs TRM:** the H-level is replaced by **32 learned latent tokens** (Perceiver/Set-Transformer style). They exchange information with the 30×30 grid-level ("L-level") stream through cross-attention. [R,C]
- **The 40.3% is pass@2 = 0.4025 on the 400 ARC-AGI-1 public *evaluation* tasks**, not the training set. [E,O]
  - All 400 IDs in the shipped `submission.json` are evaluation IDs; 0 are training IDs. Our re-scoring finds 160/400 tasks fully correct.
  - The leaderboard's "Public Train" label is therefore wrong. At most it hints that, under TRM's protocol, the eval tasks' demo pairs are training data.
- **Cost:** $2.79 is the **total inference cost for 400 tasks (≈$0.007/task)** on one L40S at $0.86/h. **Training compute is excluded.** [E]
- **The author's own conclusion:** ARM converges faster than TRM but hits the same ceiling. [R]

## Method
**Architecture [R,C]** (`utils/models/recursive_reasoning/trm_abstraction.py`).
- **L-level:** 2 transformer layers, hidden size 512, 8 heads, RoPE. It sees 900 grid tokens plus 16 puzzle-embedding tokens. Puzzle embeddings are 512-d, sparse, and trained with SignSGD at lr 1e-2.
- **H-level:** 32 latent tokens, 256-d, with no positional encoding. One H-cycle runs:
  1. H→L "broadcast": L tokens cross-attend to the latents. An optional gate controls how this enters the L input.
  2. L self-attention, repeated `L_cycles` times.
  3. L→H "perceive": the latents cross-attend to L.
  4. H self-attention, repeated `H_layers` times.
- **Halting/loss:** ACT on the mean-pooled latent (≤16 steps); stablemax cross-entropy.
- **The scored checkpoint is `abl-v1-noh`** [C]:
  - **`H_layers: 0`**, i.e. no H self-attention. This is an ablation; the README describes 3 H layers.
  - H_cycles 2, L_cycles 4, gate on, EMA 0.999.
  - batch 80, lr 1e-4, AdamATan2.
  - 700k steps (≈56M samples).

**Data [C].** The data path is `data/arc1concept-aug-1000`. It holds ARC-AGI-1 training + evaluation tasks plus ConceptARC.
- Each task gets 1,000 augmentations: dihedral transforms × color permutations, plus random translation in the train split.
- **Under TRM's standard protocol**, the eval tasks' demo pairs are trained on, which is how their puzzle embeddings get learned. Only their test outputs are held out. This is transductive test-time training, folded into pretraining.
- **The shipped `build_arc_dataset.py` is modified** to drop demo pairs of held-out puzzles (*"Demo pairs of held-out puzzles must not enter training"*).
  - If this builder produced the scored data, the eval puzzles' embeddings would be untrained, and 40% from test inputs alone would be extraordinary.
  - The dataset is not shipped, so we cannot tell which protocol produced the number.

**Inference [C]** (`utils/evaluators/arc.py`).
1. Predict on every augmented copy of the test input.
2. Invert the augmentation.
3. Hash the predicted grids and vote by count, breaking ties by mean q_halt.
4. Submit the top-2. Credit is per test pair, averaged within each task.

## Results [E]
| Checkpoint | Steps | pass@1 | pass@2 | pass@1000 | eval h | $ total |
|---|---|---|---|---|---|---|
| ARM `abl-v1-noh` | 700k | 36.4 | **40.25** | 52.4 | 2.54 | 2.18 |
| same run | 488k | 31.0 | 35.9 | 49.6 | – | – |
| TRM-opt `abl-c3-trmopt` | 488k | 18.0 | 20.75 | 35.3 | 4.61 | 3.97 |

- **Cost mismatch:** the leaderboard's $2.79 matches neither cost file. It implies about 3.2 h of eval, so it probably comes from another eval run.
- **Inconsistent "ceiling":**
  - The README claims "+~4.5 pp over TRM on ARC/pass@1 at 32M samples", and says all variants converge to "~25.0%" at 58M.
  - 25% matches `all/exact_accuracy` (0.258), not ARC/pass@1 (0.364).
- **Comparison:** the published TRM reports ~45% on ARC-AGI-1 and ~8% on ARC-AGI-2, so ARM's 40.3% is below it.
- No ARC-AGI-2 result is shipped.

## Cost / compute
- **Inference:** $0.005–0.007/task on one L40S, i.e. 2.5 h for 400 tasks with batch 64.
- **Training:** not reported (700k steps × batch 80).

## What's reusable for us
**(a) DeepSeek harness.** Nothing directly: there are no prompts and no LLM. API calls/task = 0, so cost at $0.035/call = $0.

**(b) Cross-task memory / distillation.**
- The only "memory" is the per-puzzle embedding, a learned task code. **There is no program library to transfer.**
- **Reusable:** the augmentation-voting evaluator (dihedral × color permutation, count + confidence). It is a cheap, proven test-time ensemble for the direct-output mode of our distilled model.

**(c) Kaggle (4×L4, 12 h, 240 tasks).**
- **Size:** 10.8M params is trivial to fit. Inference for 240 tasks is our guess of about 1–2 h on 4×L4.
- **Catch:** hidden tasks need their puzzle embeddings, and ideally the weights, trained on their demos *inside* the 12 h window. TRM-style pretraining took ≈56M samples.
- ARM's faster convergence matters only for such in-competition training.
- **Ceiling:** this model family scores ~8% on ARC-AGI-2 (TRM), against our 63–78% API baseline. **At best it is a cheap ensemble member or fallback, not a core component.**

## Open questions
- Was 40.3% produced with the eval tasks' demo pairs in training (the standard TRM protocol), or with the shipped leak-free builder?
- Where does $2.79 come from, and what was the training cost in GPU-hours?
- Why was an `H_layers: 0` ablation submitted rather than the 3-layer model in the README?
- What does it score on ARC-AGI-2? What are the seeds and variance?
