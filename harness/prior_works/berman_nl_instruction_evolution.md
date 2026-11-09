# Berman: Evolutionary Test-Time Compute with NL Instructions

- **Author:** Jeremy Berman. Leaderboard date 16 Sep 2025.
- **Write-up:** https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again
- **Code:** https://github.com/jerber/arc-lang-public, in `src/run.py`, `src/main.py` and `src/configs/grok_configs.py`.
  - Read at HEAD `3d820fc`; the loop is unchanged since release `6bbbc7a`.
- **License:** **none.** There is no LICENSE file, so all rights are reserved.

Source tags: [W] = write-up, [C] = code, [L] = leaderboard.

## TL;DR
- **Method:** the 2024 evolutionary loop with **English instructions in place of Python**. Grok 4 writes the instructions, Grok 4 "sub-agents" apply them to the train inputs, the score is cell accuracy, and the best instructions are revised. [W,C]
- **Results:** **29.4% on the ARC-AGI-2 semi-private set at $30.40/task.** The leaderboard's $3,648 is the **total** (120 × $30.40). [L] It also gets 79.6% on ARC-AGI-1 at $8.42/task. [W]
- Final answers are LLM-predicted grids, never verified.

## Method
**Generation (`INTUITIVE_PROMPT`)** [C]
- The prompt shows the train pairs and the test inputs as space-separated digit rows. Images are off in every production config. All calls use structured JSON.
- Excerpt: *"Apply consistently to ALL training examples… Describe the pattern without referencing specific example numbers or positions… these puzzles are designed to have elegant, intuitive solutions."*

**Executor (`AGENT_FOLLOW_INSTRUCTIONS_PROMPT`)** [C]
- A separate LLM call that returns `{"grid": …}`.
- Excerpt: *"Apply the given instructions precisely… The training examples demonstrate how the instructions work - use them to understand the pattern."*

**Fitness: leave-one-out** [C]
- For each train pair *i*, the executor gets the instructions, the other train pairs and input *i*.
- Score per pair: 1 if exact, else the fraction of matching cells (0 on a shape mismatch). Fitness is the mean.
- **Caveat:** the executor sees k−1 solved examples and may infer the rule itself, so fitness rewards instructions *plus* in-context induction.

**Individual revision (`REVISION_PROMPT`)**
- Mechanics [C]: a multi-turn chat ending in *"Your previous instructions… did not produce the correct output grids…"*, with a diff table (`actual→expected`) for each failed output. The schema requires `reasoning_for_why_old_instructions_are_wrong` before `revised_instructions`.
- Usage: individual revision is **disabled in the production config.** [C]

**Pooled revision (`SYNTHESIS_PROMPT`)** [C]
- Prompt: *"Multiple expert puzzle solvers have attempted… Each attempt captured some aspects correctly but failed in other ways… synthesize a complete, correct set of instructions."*
- It sees the top-k instructions with only the per-example percentage correct, not the outputs. `PROMPT_MECHANICS.md` claims otherwise.

**Loop** [C]
- Steps run in order, each in parallel, merging into a sorted population. **Early exit** when the best score is 1.0.
- `grok_config_prod` (unchanged since 5 Aug): `Step(5)`, `Step(5)`, `Step(20)`, then `StepRevisionPool(top5→5)`, so **at most 35 instructions.**
- The write-up describes 30 initial + 5 individual + 5 pooled = **40**. [W]

**Final answer** [C]
- If some instructions are perfect, apply up to 5 of them (cycling through) to each test input.
- Otherwise, apply the top instruction 3 times and the second 2 times, with the note *"If you think there is an error with the instructions… ignore that part"*.
- Attempt 1 is the first grid; attempt 2 is the first grid that differs from it. **There is no voting.**

**Models** [C]
- Grok 4 does every role. The later GPT-5.2 and Gemini configs shrink the search to 2+2+3 initial instructions plus a pool of 2.

## Results & ablations
- 29.4% on ARC-AGI-2 vs a previous best of 25%. [W] By our recollection, which we did not re-check, ARC Prize lists Grok 4 alone at about 16%.
- **No public-eval numbers and no quantitative ablations.**
- Qualitative: pooling more than 2 instructions can exceed token limits and at times *"degrades reasoning"*. [W]

## Predecessor (2024): Python evolution
Sources: write-up and leaderboard (18 Dec 2024). The Kaggle notebook could not be fetched.
- **Result:** Sonnet 3.5 writes `transform` functions; **53.6% on the ARC-AGI-1 semi-private set.** The $2,900 is presumably the total.
- **Track A:** 50 functions, then 200 more (easy tasks exit after 50). Then top 10 × 10 children, top 5 × 5, top 5 × 5.
- **Track B:** for each example, pool the 3 functions that solve it best. At most 500 functions per task.
- **Fitness:** perfect examples first, then correct cells. The 2 best functions give the final answers.
- **Ablation (60 train tasks, 200 calls each):** one generation scored 70%, four generations 75%.

## Reusable for us
**(a) DeepSeek port**
- `_get_next_structure_deepseek` already uses JSON mode. Changes needed:
  - switch the model to `deepseek-flash`;
  - raise the hardcoded `max_tokens` of 32k;
  - **swap the digit rows for JSON grids**, which gave us +11 points.
- **Calls.** Each instruction costs about 4 calls (1 to generate it plus 2.98 leave-one-out follows). The final step adds about 7 more (5 × 1.43 test inputs).

  | Exit after | Calls | $/task at $0.035 |
  |---|---|---|
  | step 1 | 27 | 0.95 |
  | step 2 | 47 | 1.65 |
  | full (35) | 147 | 5.15 |

- Our guess is $2–4/task on average, 30–60× the 2-call baseline.
- Cheap upgrade: majority-vote over the final follows.

**(b) Distillation.**
- Two clean roles: the **instruction writer** (hard) and the **executor** (narrow, highly distillable).
- The leave-one-out logs give graded (task, rule, score) labels and revision pairs that improved the score.
- The NL rules can serve as CoT targets.

**(c) Kaggle (4×L4, 12 h).**
- Assume ~1.5k tok/s per L4 for an 8B AWQ model (our assumption). That is ~1.1M tokens per task, so 150 calls would need ≤7k tokens each.
- A trimmed version is needed: about 10 instructions with leave-one-out, ~45 calls of ~20k tokens.
- The missing license means we should **reimplement** (about 400 lines).

## Open questions
- Which config produced 29.4% (35 or 40 instructions)? Runs and variance?
- How much does the executor's own induction contribute?
- Cost spread across tasks; how often early exit fires.
- Does NL evolution help a base model already at 78% pass@2?
- Does DeepSeek's JSON mode work together with max-effort thinking?
