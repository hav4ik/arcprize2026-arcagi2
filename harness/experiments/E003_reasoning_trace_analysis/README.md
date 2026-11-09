# E003: Reasoning-trace analysis, DeepSeek Flash vs Pro

## Question
How do DeepSeek V4.1-Flash and V4-Pro think on ARC-AGI-2? Where do the tokens go, why do they fail, and what should a harness or distilled model keep or fix?

## Setup
- **Tasks:** 4 ARC-AGI-2 public-eval tasks with one test input each, chosen by Flash's earlier E001d result:
  - `db695cfb` (solved on both attempts)
  - `8b9c3697` (solved on one)
  - `2b83f449` and `5545f144` (solved on neither)
- **Calls:** ARC Prize prompt, max effort, 2 attempts per model, 16 calls in all, ~$1.6.
- **Command:**
  ```bash
  cd evals && ../.venv/bin/python run_deepseek_arcprize.py --model <m> --effort max --tasks db695cfb,8b9c3697,2b83f449,5545f144 --tag traces
  ```
- **Data:**
  - Records: `evals/results/arcprize_deepseek-{flash,v4-pro}_max_traces.jsonl`
  - Full reasoning: `evals/results/arcprize_deepseek-*_max_traces/*.reasoning.txt.gz` (60k–280k chars each)
- **Analysis:** two agents each read the full traces for half the tasks.
  - `part1_successes.md`: db695cfb, 8b9c3697
  - `part2_failures.md`: 2b83f449, 5545f144

## Results

| Task | Flash | Pro | What the rule needs |
|---|---|---|---|
| db695cfb | 2/2 | 2/2 | Diagonal geometry; easy |
| 8b9c3697 | 2/2 | 2/2 | Exact selection rule found through counterexamples |
| 2b83f449 | 1/2 | 0/2 | Conservation: the count of 3s per connected region stays constant |
| 5545f144 | 0/2 | 0/2 | Train panels are animation frames; simulate a 12-step walk |

Flash: 5/8 correct; Pro: 4/8. Flash uses 0.5–0.7× Pro's tokens and is 3–5× faster.

## Patterns in the thinking

1. **The rule is often found early, and the budget goes elsewhere.**
   - On the easy task, the rule appears within the first 7–13% of the trace.
   - Building the output grid by hand and counting cells takes 25–45%.
   - Second-guessing after the answer is done takes up to 60% (Pro).
2. **Verification is weak and object-level.**
   - No trace re-renders a train output cell by cell.
   - One Flash attempt never looked at one of the train pairs.
   - The correct 2b83f449 trace spent 22% of its tokens on verification; the failures spent ≤10%.
3. **Every wrong attempt noticed a contradiction and explained it away.** Examples:
   - an invented "deficit" patch;
   - thresholds fitted to each pair separately;
   - "Let's not over-stress about count" — said right after noticing the fix that would have made the answer correct.
4. **The right idea often appears and is dropped.**
   - All four 5545f144 traces said "panels are animation frames".
   - One Flash attempt confirmed the movement rule on one sequence, then abandoned it when it didn't fit another pair.
5. **Right answers can come from wrong rules.** Flash reached correct 8b9c3697 outputs with a loose approximation of the rule in one attempt and by analogy to a train pair in the other. Correctness labels alone overstate reasoning quality.
6. **"Given the time…" comes right before a guess.** Yet no trace hit a limit: all stopped normally at 61–88k tokens. The model gives up on its own.
7. **Perception is mostly fine.**
   - Grids are handled in (row, col) coordinates.
   - About 5–6 small misreads in the successful traces, all self-corrected within ~1k chars.
   - Two misreads in the Flash failures were not caught.
8. **Flash vs Pro.**
   - Pro hedges heavily (334–438 "maybe"s per trace).
   - Pro is more systematic, but on the hard failures it carefully checks fitted thresholds rather than the real rule.
   - Pro does not find rules sooner, and it is not better on this sample.

## Implications

**Harness (E002):**
- **Required transform function.** Require a runnable `transform()` checked on all train pairs. Mechanical execution replaces hand construction and counting (25–45% of tokens). It would have rejected 3 of the 4 wrong 5545f144 rules, Flash's loose 8b9c3697 rules, and the skipped train pair.
- **Invariant checks on the test answer** (counts per colour or per region, as seen in the train outputs). These would have flagged all 3 wrong 2b83f449 attempts.
- **Grid and panel extraction tools** (objects, connected components, diffs) would have prevented the uncaught misreads.
- **Second opinion.** A second-opinion or refinement call fed the trace's own scattered observations ("panels are frames…") targets the failure mode where the right idea was dropped.
- **Estimated saving** (not measured): 50–70% fewer tokens on solved tasks.

**Distillation:**
- **Keep:**
  - coordinate inventories;
  - hypothesis → counterexample → refine tables;
  - "explain every distractor" checks;
  - building the output row by row from the rule;
  - Flash's terse style.
- **Prune:**
  - enumerating cells one by one and re-listing rows;
  - hedging;
  - alternatives with no supporting evidence;
  - long unproductive openings (the correct 2b83f449 trace wasted its first 57k chars).
- **Labelling:** label traces by whether the rule is valid on all train pairs, not by the final answer alone. Exclude near-miss failures, because their key moves are rationalisations.

## Caveats
Only 4 tasks and 16 traces, qualitative analysis. Token-saving numbers are estimates. Next step: measure these patterns on a larger sample, using the trace-saving scripts, which now save every call's reasoning.
