# Imbue: Beating ARC-AGI-2 with Code Evolution

- **Authors:** Daniel Mewes, Nayana Bannur, Catherine Kim (Imbue). Published 27 Feb 2026, updated 14 May 2026.
- **Blog:** https://imbue.com/blog/2026-02-27-arc-agi-2-evolution (framework post: https://imbue.com/research/2026-02-27-darwinian-evolver/)
- **Code:** https://github.com/imbue-ai/darwinian_evolver, in `problems/arc_agi.py` and `imbue_experiments/run_arc.py`.
  - Repro commit is `1a50d69`; the Gemini 3 Flash run used `482ac63`.
  - `arc_agi.py` has not changed since `1a50d69`.
- **Paper:** none
- **License:** AGPL-3.0. The helper file `arc_agi_poetiq.py` is Poetiq's MIT code.

Source tags: [B] = blog, [C] = code (at `1a50d69`).

## TL;DR
- **Method:** a separate evolutionary search runs for each task. Each candidate is a Python `transform(grid)` program, and an LLM rewrites a parent after seeing feedback on the train pairs. [B,C]
- **Fitness:** 0.9 × train correctness + 0.07 × an LLM judge of how well the solution transfers to the test inputs + 0.03 × code simplicity. [B,C]
- **Results:** ARC-AGI-2 public eval, pass@2 [B]:
  - Kimi K2.5: 12.1% → **34.0%** at $2.67/task
  - Gemini 3 Flash: 34.0% → **61.4%** at $2.42/task
  - Gemini 3.1 Pro: 88.1% → **95.1%** at $8.71/task
- Imbue used the same framework and prompts for all three models. There is no semi-private score because that evaluation was "on hold". [B]

## Method
**Representation.** Each organism holds code plus `from_explanation`, the LLM's natural-language rule. The root organism returns the input unchanged. Code runs in a subprocess with an 8 GB memory limit and a 30 s timeout. [C]

**Loop [C].** Settings: sharpness 10, midpoint percentile 99, novelty weight 0.2, verification on, 2 parents per iteration.
1. Sample 2 parents with replacement, with weight `sigmoid(10·(score − p99_pop)) / (1 + 0.2·n_children)`. Organisms that already pass all train pairs can still be parents.
2. Run the main mutator (high thinking) on each parent.
3. With probability 0.25, and only once the population has at least 3 organisms, also run **crossover**. Crossover takes 3 parents, sampled without replacement at novelty weight 1.0.
4. **Verification:** discard a child unless its output differs from its parent's on at least one train or test input.
5. Score the children and add them. The population is never pruned.
6. Stop on one of three conditions:
   - 16 iterations;
   - 5 hours;
   - a solution that is correct on train and has transfer ≥ τ, plus either (all train-perfect solutions agree on the test outputs) or (≥2 distinct high-transfer predictions per test and two high-transfer solutions that agree).

   τ is 0.95 for Gemini and Kimi, 0.7 for GPT-5.2 and 0.35 for Opus.

**Mutation prompt [C]** (adapted from Poetiq). It asks for natural-language reasoning first:
- *"Formulate a Hypothesis BASED ON THE NATURAL LANGUAGE"*
- *"**AVOID EDGE CASE THINKING**: ARC tasks have a unified solution that works for all examples"*
- *"Make sure that the transformation step can also be applied to the challenge inputs"*

The response must contain a `## Transformation Steps` section followed by one python block.

The feedback the mutator sees includes:
- for each example, right or wrong, with a diff grid in `pred/correct` format, the accuracy, and any traceback;
- the parent's **challenge outputs**, labelled "we do not know whether this prediction is correct".

When the parent is not yet perfect on train, a 50/50 choice picks the guidance:
- NORMAL: *"First see if you can adapt and/or extend the current solution…"*
- AGGRESSIVE: *"IMPORTANT: You have already tried to fix this solution several times and it hasn't worked! Please go back to the drawing board…"*

When the parent is already perfect on train, the prompt deliberately misleads the model: *"it does not yet generalize to the challenge inputs"*.

When input and output are similar (cell similarity ≥ 0.7), the prompt adds an `in/out` diff grid for each example. [C]

**Fitness [B,C]:** `0.9·correctness + 0.07·transfer + 0.03·simplicity`
- **Correctness:** mean cell accuracy (0 if the shape is wrong), rescaled so the identity function scores 0.2, clamped at 0, then `0.9·x + 0.1·[all train correct]`.
- **Transfer:** computed only for organisms perfect on train. An LLM receives the problem, the explanation and the test outputs, and scores `applies_to_challenges`, `correctly_applied_pattern` and `consistent_with_examples`. The transfer score is the product of the three. The prompt says *"Be VERY RIGOROUS!"*.
- **Simplicity:** a low-thinking LLM counts literals and hard-coded colors. The score is `0.2·lit + 0.4·color + 0.4·(1−1/(1+len/3000))`.
  - **Discrepancy:** as written, this term rewards *longer* code, while the blog says shorter is better.

**Final 2 attempts [C].** Attempt 1 is the output of the top-scoring organism. Attempt 2 is chosen separately for each test input: the highest-scoring organism whose output differs. There is no voting.

**Models [C].**
- 3.1 Pro run: high and medium calls go to 3.1 Pro; the simplicity call goes to 3 Flash.
- Flash run: everything goes to 3 Flash.
- Kimi run: everything goes to Kimi via OpenRouter.
- **Gemini high-thinking calls have server-side code execution enabled.** The learning log is off.

## Results & ablations [B]
| Model | Base | Evolved | $/task |
|---|---|---|---|
| Kimi K2.5 | 12.1% | 34.0% | 2.67 |
| Gemini 3 Flash (High) | 34.0% | 61.4% | 2.42 |
| Gemini 3.1 Pro | 88.1% | 95.1% | 8.71 |

- From the blog's chart, the base runs cost about $0.30, $0.23 and $1 per task. Evolution therefore costs roughly 9–10× the base.
- For comparison, the blog cites Confluence Labs at 97.9% for $11.77/task.
- **There are no quantitative ablations.** The qualitative claims are:
  - using natural language alone as the representation gave no gain and cost much more;
  - simplicity helps "slightly";
  - verification gives ">10×" better efficiency (a claim about the general evolver).

## Cost / compute
- Each task makes up to about 32 mutation calls and about 8 crossover calls. Each child adds a simplicity call, and each train-perfect child adds a transfer call. [C]
- The default task concurrency is 32. A full run takes "a few hours" and costs a "three-digit USD amount". [B]

## Reusable for us
**(a) DeepSeek harness.**
- Add a provider branch modelled on the OpenAI-compatible `_prompt_llm_openrouter`.
- Our estimate (not Imbue's): with the 9–10× multiplier, V4.1-Flash would cost about **$0.2–0.4/task**. Pro could act as the mutator with Flash as the judge.
- To recalibrate:
  - τ;
  - a replacement for Gemini's code execution, such as local exec feedback or tool use;
  - the sign of the simplicity length term.

**(b) Data generator.**
- The JSONL logs hold code, explanation, lineage and scores. Three useful datasets can be mined from them:
  - (task → NL steps + program) pairs from organisms that are perfect on train;
  - (parent + diff feedback → better child) repair triples;
  - transfer-judge labels checked against `gt_score`.
- Running it on ARC-1/2 training data and synthetic tasks would produce a corpus of verified programs.

**(c) Kaggle offline (4×L4, 12 h).**
- Rough budget: 120 tasks × ~50 calls × 10–20k tokens comes to 60–120M generated tokens, or about 1.4–2.8k tok/s sustained. That only works with a small or quantized local model and fewer iterations.
- Evolution multiplies base ability. Even the 1T-param Kimi reached only 34%, so a ~30B local model would start far lower.
- AGPL may conflict with ARC Prize open-source terms (to be checked). Reimplementing it is safer.

## Open questions
- Number of runs, variance and seeds. No semi-private score.
- How much each component contributes (transfer, simplicity, crossover, aggressive prompt), and how much Gemini's code execution adds.
- How iterations and cost are distributed across tasks, and how often early stopping fires.
- The exact hyperparameters for the Kimi and 3.1 Pro runs. The blog points to `1a50d69`, which defaults to p99, 2 parents and 0 extra iterations; the Flash run used p50 and 2 extra iterations.
- Whether the sign of the simplicity length term is intended.
