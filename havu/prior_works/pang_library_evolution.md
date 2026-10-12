# Eric Pang: Efficient Evolutionary Program Synthesis

- **Author:** Eric Pang. Leaderboard date 1 Sep 2025; write-up 16 Sep 2025.
- **Write-up:** https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary (also mirrored in the repo README).
- **Code:** https://github.com/epang080516/arc_agi, HEAD `2b1a3c1`, built on Jeremy Berman's 2024 `arc_agi` repo (prompts via Greenblatt).
- **License:** none (all rights reserved). The vendored `lpn/` is Apache-2.0.

Source tags: [W] = write-up, [C] = code, [O] = our analysis of the shipped pickles.

## TL;DR
- **Method:** DreamCoder-style, but with no DSL. [W,C]
  - Grok-4 writes Python `transform` programs.
  - The best program for each task is appended to one global library.
  - For each new task, 2 library programs are sampled by how well they fit the train pairs and are shown in the prompt as a "primitive".
- **Budget:** a seeding pass over the 1,000 ARC-AGI-2 training tasks (1 call each) produced 538 programs. Eval then makes **≤10 calls/task**. [W,C]
- **Results:**
  - **ARC-AGI-2 semi-private: 26.0%.** The leaderboard's $476.4 is the **total for 120 tasks = $3.97/task** (same convention as Berman's $3,648).
  - ARC-AGI-1 semi-private: 77.1% at $2.56/task. [W]
- **Caveat:** in the shipped code, retrieval is nearly uniform, so the library's value is unproven. [O]

## Method
**Base prompt [C]** (Greenblatt's `REASONING` prompt).
- Key lines:
  - *"your implementation in code must be exactly correct"*
  - *"Consider explicitly asking yourself (in writing): 'Are there any additional parts or exceptions to the transformation rule that I might have missed?'"*
  - *"Start your response by carefully reasoning in <reasoning></reasoning> tags"*
- **One-shot:** one hand-written solved example (ARC-1 task `1f642eb9`, or `4be741c5` when the output shape changes).
- **Grid formats:** each grid is given as its shape, a PNG, `|`-separated ASCII and a Python list. Same-shape tasks also get an input→output diff grid.

**Library injection [C].** Each sampled program becomes an extra user message:
> "Here is a primitive that you can build upon to solve this problem. Remember, it's okay to not to use this primitive too if you think it's not helpful. {code} Here are the examples that have failed: {diff_str}"

For failed train pairs whose output shape matches, `diff_str` gives the input, the wrong output, the expected output and a cell diff.

**How the library is built, retrieved and scored [C].**
- **Entry:** a whole `transform` function; no NL description, embedding or refactoring. The shipped `saved_library_1000.pkl` has 538 entries with a median of 35 lines. [O]
- **Admission:**
  - After each task, attempts are ranked by train accuracy (the fraction of pairs that are exact), with mean cell accuracy as the tie-break.
  - The top-1 is appended **unconditionally**, even if it is wrong. A TODO says "only add primitive if top one scores better…".
- **Retrieval:**
  - Every library program is run on the new task's train inputs (5 s timeout).
  - Score = number of exact pairs + cell accuracy.
  - Softmax at T=1, then draw k=2 without replacement.
- **Design choices [W]:** tuned on 40 ARC-1 training tasks: Grok-4, failure diffs ("significantly improves"), softmax over greedy, k=2. No numbers given.
- **LPN [W]:** an LPN-based selector was "promising" but exceeded Kaggle's 12 h limit.

**Loop and budget [C]** (`grok_dreamcoder_tree`).
- **Calls:** 5 parallel `grok-4-0709` samples, temperature 0.95, `max_tokens` 120k, **`max_retries=0`** (hence 538 programs from 1,000 tasks).
- **Repair:** Berman's fix tree is commented out.
- **Rounds:** eval makes 2 passes over all tasks. Round 2 skips a task if its top-2 programs are both train-perfect.
- **Library growth:** it also grows during eval, so round 2 can retrieve a task's own round-1 best.

**Final answers [C].** The 2 attempts are the top-2 programs with distinct outputs by train accuracy, kept across rounds. There is no voting.

**Transfer [O].** `challenge_primitive_accuracy_scores.pkl` scores the 120 ARC-AGI-2 public-eval tasks against the 538 training-derived programs.
- **No task** has a program that solves all its train pairs. Only 5 tasks have one that solves even one pair.
- The softmax is nearly flat: the best program's median sampling probability is 0.27%, and the best/median weight ratio is 1.5.
- **Retrieved "primitives" are therefore effectively random.** The gain over Grok-4 alone (16%) is likely driven by:
  - 10 samples;
  - train-accuracy selection;
  - round-2 self-feedback.

## Results [W]
| Set | Score | $/task |
|---|---|---|
| ARC-AGI-2 semi-private | 26.0% | 3.97 ($476.4/120) |
| ARC-AGI-1 semi-private | 77.1% | 2.56 |

- **Leakage risk:** ARC-AGI-2 training (the library source) contains ARC-AGI-1 public tasks, so `-v1 -e` on ARC-AGI-1 public eval would leak.
- There is no library ablation and only one run.

## Cost / compute
- About 10 Grok-4 calls/task ($3/$15 per M tokens), roughly $0.40/call. Retrieval is CPU-only and negligible.

## Reusable for us
**(a) DeepSeek port.**
- Swap the xAI client for OpenAI-compatible; drop images and the one-shot; use JSON grids (+11 pts for us).
- **Our estimates:**
  - building the library: 1,000 calls ≈ **$35, one-time**;
  - eval: ≤10 calls ≈ **≤$0.35/task** (about $42 for 120 tasks), against $0.07/task for our baseline.
- Sharpen retrieval (T≈0.1 or greedy top-k) and admit only train-perfect programs.
- **First experiment, which Pang never ran:** compare 10 code samples + train verification, with and without library hints, against our 78.1% direct-output baseline.

**(b) Library as cross-task memory and distillation data.**
- **SFT data:** train-perfect programs on training and synthetic tasks give verified (task → reasoning → program) pairs. Primitive + diff → better program gives repair pairs.
- **Offline retrieval:** retrieval by execution needs no GPU, so a pickled library can feed hints to a small model on Kaggle.
- **Upgrades:** NL summaries, sub-function extraction, dedup.

**(c) Kaggle (4×L4, 12 h, 240 tasks).**
- 2,400 generations × ~50k tokens = 120M tokens, or ~2.8k tok/s sustained. That needs a distilled model with ~5–10k-token outputs, or fewer samples.
- No license, so we should reimplement it (about 200 lines of logic).

## Open questions
- Does the library beat plain resampling? There are no seeds or variance figures.
- Does the $476.4 include the library build?
- Was the same T=1 retrieval used on semi-private? How often does round 2 retrieve the task's own program?
