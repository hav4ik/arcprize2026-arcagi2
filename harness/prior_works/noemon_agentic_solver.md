# Noemon: Agentic ARC-AGI-2 Solver (Reasoner → Validator → Refine)

- **Authors:** Noemon (company). Leaderboard date 2026-04-09; repo HEAD `54c7c3a` (12 Apr 2026).
- **Site:** https://noemon.com (teaser only). **No blog or paper.**
- **Code:** https://github.com/noemon/noemon-arc-agi (fork of `arcprize/arc-agi-benchmarking`); solver in `src/arc_agi_benchmarking/noemon/` (`prompt_bundle.md`, `batch_engine.py`).
- **License:** MIT.

Source tags: [S] = site/README/plot, [C] = code.

## TL;DR
- **Method:** a natural-language-only loop with **no code execution**. [C]
  - A high-thinking **Reasoner** writes an "instruction set": the rule in prose.
  - A medium-thinking **Validator** applies the rule to held-out train pairs.
  - The Reasoner revises for up to 3 rounds.
  - Extra solvers add candidates. Stop when **any grid appears twice**; otherwise a **judge** picks.
- **Result:** **92.5% on ARC-AGI-2 public eval, ~$3.9/task**, Gemini 3.1 Pro, self-reported. [S]
- **The "instruction set" is per task.** Nothing persists across tasks. [C]

## Method [C]
**Input format.**
- Grids are compact JSON, one row per line, plus `grid_info` (shape and colour counts).
- The Reasoner sees train pairs up to 3000 cells. Any overflow arrives in round 2 as "additional training pairs".

**Reasoner** (thinking high, temperature 1.0). The prompt requires:
- *"Do not assume any training pair is incorrect"*
- *"stated only in terms of properties derivable from the input grid… Do not refer to specific examples"*
- *"Every part of the output must be fully implied by the instructions"*

It prefers *"more elegant/general explanations over ad hoc pixel-level descriptions"*.

The output is JSON `{instructions}` with these sections: `High-Level Summary`, `Grid Size`, `Invariants`, `Transformation Algorithm`, and an optional `Rejected Alternative` (*"trap that the validator should avoid"*).

**Validator** (thinking medium, temperature 0.7). The prompt says *"Carefully follow `instructions`… Do not overthink… do not invent steps or patch the rule."*
- It works leave-one-out: instructions plus the other train pairs (up to 2000 cells), then predict the target pair's output.
- It probes **2 random pairs** first, and runs the rest only if both pass.
- Pairs solved in earlier rounds are skipped.
- Feedback is either "correctly solved", a `cell_diff_grid` (`expected/predicted`), or the predicted grid when the shape is wrong.

**Refinement.** Round 2 continues the same chat with the feedback.
- If every checked pair failed: *"You may have to completely revise your hypothesis"*.
- Otherwise: *"pay attention to the detail… small amendment"*.
- Both add *"Prefer rules a human would perceive as more elegant… physical intuition… make up a story of how objects move"*.

**Solving.**
- **Train validated:** the Reasoner solves the test input and a test-Validator call applies the instructions independently. A match stops the attempt.
- **Still failing after round 2:** round 3 solves anyway (*"validator has lower reasoning and sometimes fails"*).

**Thought history.** The Reasoner is one multi-turn chat; raw Gemini responses, including `thoughtSignature` parts (encrypted reasoning state), are appended each round.

**Final solvers** (iterations 4–6; thinking high, temperature 0.7) are independent calls.
- Train validated: they get the instructions plus **one rotating train pair**.
- Otherwise: they get all pairs plus the last instructions and feedback.

**Self-consistency and judge.**
- After each iteration, the attempt stops if any two candidates are identical.
- Otherwise the **judge** (thinking high, temperature 0) sees all train pairs, the test input, the instructions (*"hint, not as ground truth"*) and lettered candidates with provenance; it returns `candidate_letter`.

**Other details.**
- Every role is forced to JSON through Gemini `response_schema`.
- The 2 attempts are **independent full runs**, with nothing forcing them to differ.
- The `reasoner_test_validator_max_rounds` prompt appears unreachable in the batch loop.

**Calls per attempt:**
- Best case: Reasoner + n_train validators + solve + test-Validator, about **6 calls**.
- Worst case: 3 Reasoner + about 4 validators + 4 solvers + judge, about **12 calls**.

## Results
| Set | Model | Score | $/task |
|---|---|---|---|
| ARC-AGI-2 **public eval** (120 tasks) | Gemini 3.1 Pro Preview | 92.5% | ~3.9 |

- **Which set:** README, site and plot ("ARC-AGI-2 LEADERBOARD (public eval)", "all 120 tasks") all say **public eval**, not training; the leaderboard's "Public Train" looks like a mislabel. [S]
  - Caveat: the README repro command clones `fchollet/ARC-AGI` (ARC-AGI-1), inherited from upstream.
  - The public eval is not train-contaminated, but it has been public since March 2025.
- **Cost.** $3.9 is **per task** at full non-batch pricing ($2/$12 per 1M tokens). Real batch spend is probably about half.
- **No ablations, seeds or variance.**
- **Comparison.** Imbue measured base 3.1 Pro at 88.1% (~$1/task): Noemon adds ~4 points at ~4× cost; Imbue reached 95.1% at $8.71.

## Reusable for us
**(a) DeepSeek harness.**
- **Cost estimate (ours):** 6–12 calls × 2 attempts × about 1.4 test inputs per task gives **about 17–34 calls/task, or $0.6–1.2/task** at $0.035/call. Our baseline costs $0.07.
- **Prompts** port directly; the grid format matches our better JSON format.
- **Thought history:** DeepSeek doesn't take prior `reasoning_content` back; drop it or inline a summary.
- **Output parsing:** use JSON mode, or the existing last-JSON-blob fallback parser.
- **Ablations to run:**
  - probe validation;
  - the duplicate-stop rule;
  - forcing attempt 2 to differ.

**(b) Distillation.** Every role is single-step text→JSON, which makes clean SFT data:
- (task → instructions that pass leave-one-out);
- (instructions + input → grid), with free ground truth on train pairs;
- (feedback → revision);
- judge labels.

Generate on ARC-1/2 training and RE-ARC/BARC. Weak link: a small model *executing* prose rules on large grids.

**(c) Kaggle (4×L4, 12 h, 240 tasks).**
- Load: about 5.4k calls × 15–20k tokens ≈ 80–110M tokens, or about 2–2.5k tok/s sustained.
- Feasible only with a ~8B model, short thinking, and a round-1 rule shared across attempts.
- Leave-one-out validation is a free confidence signal for gating compute.

## License
- MIT, so the prompts and loop can be copied with attribution.
- To check: whether ARC Prize 2026 demands CC0/MIT-0.

## Open questions
- How often each exit (validated, duplicate, judge) fires; tokens per role.
- Value of thought signatures vs plain re-prompting.
- Variance, and a semi-private score.
- Whether the dead prompt was live in the scored run.
- Leaderboard "Public Train" label unverified (page is JS-rendered).
