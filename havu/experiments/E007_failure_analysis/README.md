# E007: Failure analysis of E001d wrong answers (no API cost)

## Question
When DeepSeek V4.1-Flash (max effort, ARC Prize prompt, no tools) gets an ARC-AGI-2 pair wrong, did it miss the idea, or did it have the rule and fail the machinery of executing it by hand?

## Setup
- **Data:** the 117 wrong attempts in E001d (`evals/results/arcprize_deepseek-flash_max.jsonl`), across 68 tasks. The 17 API errors are excluded.
- **No reasoning traces:** E001d saved only final grids. Every verdict comes from comparing the prediction with the truth, the train pairs and the sibling attempt.
- **Exceptions:** 2b83f449, 5545f144 and 8b9c3697 also have E003 trace analyses.
- **Method:** 7 Claude subagents, about 10 tasks each. For each task an agent derived the true rule, inferred the rule the model applied, and labelled each wrong attempt:

| Label | Meaning |
|---|---|
| `missed_idea` | Core concept wrong |
| `partial_rule` | Core right, but a sub-rule, condition or tie-break wrong (systematic) |
| `execution` | Rule right, hand execution wrong |
| `perception` | Input misread |
| `ambiguous` | Train pairs don't decide between readings |

- **Two flags per attempt:**
  - `code_check_would_catch`: would running the inferred rule on the train pairs expose the error?
  - `tool_would_fix`: would Python execution alone fix it?
- **Files:** per-attempt labels in `classifications.jsonl`; per-group notes in `group_notes/`. Confidence: 52 high, 54 medium, 11 low.

## Results

| Category | All wrong attempts (117) | Pair never solved (67) | Sibling attempt correct (50) | Median output tokens |
|---|---|---|---|---|
| partial_rule | 53 (45%) | 29 | 24 | 63k |
| execution | 37 (32%) | 18 | 19 | 63k |
| missed_idea | 19 (16%) | 16 | 3 | **100k** |
| ambiguous | 5 | 3 | 2 | 49k |
| perception | 2 | 0 | 2 | 41k |
| other (gave up) | 1 | 1 | 0 | 131k |

**Unsolved pairs (36 with at least one wrong attempt), classified by their closest attempt:**

| Closest attempt | Pairs |
|---|---|
| partial_rule | 15 |
| execution | 13 |
| missed_idea | 6 |
| ambiguous | 2 |

The 6 missed-idea pairs come from 5 tasks: 269e22fb, 4c7dc4dd (×2), 5545f144, 6ffbe589 and abc82100.

**What would help:**
- **Python execution** fixes 34 of the 37 execution errors and none of the partial-rule errors.
- **A train-pair check** would catch 69 of 117. That is 39 of 53 partial-rule errors, 18 of 19 missed ideas, but only 11 of 37 execution errors: hand-execution slips on the test grid usually don't show up on train.

**Near misses:**
- 36 attempts have the right shape and at most 10 wrong cells.
- 22 have the wrong shape; 15 of those are just a row added or dropped, mostly with the content otherwise right.

## Findings
1. **Missing the idea is rare.** It accounts for 16% of wrong attempts and 6 of 36 unsolved pairs. When it happens, the model searches long (median 100k tokens) and falls back to a simpler concept, such as "delete the dots" or tiling and mirroring.
2. **The main failure is a sub-rule that is wrong or missing, usually one a train pair shows.** Typical cases are a condition, an exception, a tie-break, which object binds to which role, an orientation, a stacking order, or whether rays cross. This matches E003: the model notices a contradiction and explains it away.
3. **About a third are machinery errors:**
   - dropped or added rows on large grids;
   - off-by-one when copying or extending lines;
   - geometry around concave corners (13e47133, all 4 attempts);
   - hand counting and flood fill;
   - path tracing that stops early or overshoots.
4. **The model fits the training examples' surface.** It reuses the layout or orientation of a train output, or matches an unseen test shape to the nearest train shape (78332cb0, fc7cae8d, eee78d87, 2d0172a1, e3721c99). Some of these shortcuts fit every train pair, so a train check alone can't catch them.
5. **The two attempts vary a lot.** When one attempt is right, the wrong sibling is almost always a small deviation (3/50 missed the idea). For gaps in the rule itself, the two attempts are often identical (88e364bc, d35bdbdc), and a second sample adds nothing.

## Implications for E004 (tool use)
- Code execution should recover most of the 13 unsolved pairs whose closest attempt was an execution error. It should also cut tokens, since E003 found 25–45% of tokens go to building grids by hand.
- Partial rules, the largest group, need **verification in the loop**: write `transform()`, run it on all train pairs, and return the cell diffs to the model. This would flag about 75% of them, and a revision then has a chance. The `code` variant in the tool-use spec covers this.
- The remaining hard core: about 6 missed-idea pairs, a few train-consistent shortcuts, and ambiguous pairs. Neither tools nor train checks fix these.
- **Rough ceiling (not measured):** if execution fixes most of the 13 execution pairs and verification plus revision recovers half of the 15 partial-rule pairs, pass@2 could gain on the order of 10 points over 78.1%.

## Caveats
- Labels are inferred from outputs, not traces, so the split between partial_rule and execution is a judgement call.
- Medium or low confidence on 65 of 117 labels.
- Single grader per task.
