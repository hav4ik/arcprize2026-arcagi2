# Sample C: right shape / unparsed / length cap (10 attempts)

Nemotron 3.5 Lightning, `high`, ARC Prize prompt, max_tokens 200k. Source: `sample_C_right_shape_unparsed_length.json`.
"Cells wrong" compares the prediction with the solution. "Δ vs input" counts the cells the prediction changed from the test input. Offsets are char offsets into the decompressed reasoning trace.

## Per-attempt

**b6f77b65 p0** (right shape, 44/144 wrong, 74k tok, Flash 1/3)
- Rule found only in part. It correctly removes every cell whose colour matches the (0,0) corner (7). It then applies per-cell column gravity, where the correct rule makes whole objects fall as rigid units. Wrong rule, not an execution slip.
- It had the right idea at @124.5k: "Maybe the transformation is not per-cell gravity, but per 'connected component' or 'shape' shift." It never tested that idea.
- It had already refuted per-cell gravity on Ex1 ("So not matching", @135k) and still submitted it: "I'll produce the output based on the 'remove other same-valued as top, gravity down order-preserving fall' rule."
- Tokens: roughly 15 hypotheses (shift down by 3, wrap, rotate, gravity variants), each re-derived by hand over full 12x12 grids. Its checks were accurate, but it ignored what they showed.

**d35bdbdc p0** (unparsed, finish=length, 200k tok, Flash 0/3)
- Degenerated at char **82,568** of 494,576, during a column-by-column re-transcription of the test grid: `Col8: row2=3, row3=2, ... Also row? row? row? row? ...`. The token "row? " then repeats unbroken for about 412k chars, so roughly 83% of the budget went to the loop.
- Before the loop it was already lost. It had invented a spurious positional rule, "the changed cells ... are always in column 2 or column 9!", and was transcribing the test grid with 1-based and 0-based row numbers mixed (it calls row 7 "row8").
- No answer was ever written. Verification before the loop: 3 tests in 82k chars.

**7c66cb00 p0** (right shape, 298/720 wrong, 70k tok, Flash 3/3)
- It misread the grid size from the start: it says "25x25" 13 times, but the grids are 30x24. It then miscounted band rows throughout ("Rows 1-10: all 8s ...").
- It never found the rule (the small shapes are erased and stamped into the band of matching colour). Its final grid leaves the shapes in place and shifts the bands by several rows.
- It gave up explicitly: "Given the absolute lack of time ... I give up. I'll output a placeholder." (@129k). That came at 70k of 200k tokens.
- Tokens: 27% grid transcription, and most of the rest went to per-row diffing of misread grids.

**9bbf930d p0** (right shape, 12/483 wrong = 97.5% "accuracy", 95k tok, Flash 0/3)
- The near-miss is illusory: the prediction is the test input unchanged (Δ vs input = 0). The solution changes 12 cells.
- It could not count the columns in a row ("ignoring the exact column count issues"). It cycled through row-rotation and colour-swap rules, each of which failed on Ex0 row 0.
- End: "Given the time, I'll output the test input exactly as given."

**d35bdbdc p2** (right shape, 39/100 wrong, 70k tok, Flash 3/3)
- It output an all-zero grid. Accuracy is 61% only because the grid is mostly background.
- It spent most of the trace on arithmetic rules for the centre cells ("Maybe Z = (X + Y) mod (X XOR Y + 1)?", "number of letters in the English name"). It never noticed the 5-path that connects the boxes.
- End: "Given the extreme time pressure ... I surrender. I'll output all zeros."

**dd6b8c4b p1** (right shape, 12/121 wrong, 58k tok, Flash 3/3)
- Rule found only in part. It noticed a real invariant: "the number of 9's that disappeared equaled the number of new 9's." It never worked out which 9s move (those reachable outside the 6-wall) or how the 3x3 box fills (in reading order).
- The submitted answer turns six 3s into 9s and removes no 9s. All 6 changed cells are correct, but 12 required changes are missing.
- The trace and the answer disagree. The trace ends "I'll just output the test input", but the content field holds a different grid.
- "Given the time" appears 9 times, the first at 49% of the trace.

**247ef758 p1** (right shape, 22/216 wrong, 82k tok, Flash 0/3)
- Rule found only in part. It erased the left-hand shapes (correct) and never placed them in the box at the intersection of the matching row and column markers.
- It saw that the training outputs disagreed and submitted anyway: "in the training examples, there were activated cells. However, maybe the test input is designed such that no activated cells appear".
- It used a spurious rule, "pivot column p=4 because the grid contains an 8". It also made a transcription slip that put a stray 6 at (4,4).
- It transcribed a training row with the wrong length ("(15 cols)").

**247ef758 p0** (right shape, 24/300 wrong, 72k tok, Flash 2/3)
- Same failure as p1: it erases the legend shapes and places nothing ("zero 0,1,2 ... keep the rest").
- It also chose the wrong erase width, so a 7 at (12,3) and another at (14,3) survive.
- The end of the trace is careful self-checking of the wrong rule ("Correct."). Its verification is accurate at the cell level, but it is aimed at the transcription, not at whether the rule holds on the training pairs.

**88e364bc p1** (right shape, 16/400 wrong, 62k tok, Flash 3/3)
- It gave up and meant to copy the input: "I'll output the test input. It's the only safe format-wise."
- It got even the copy wrong. Four cells differ from the input, none of them correctly. A stray 3 appears at (6,7), and row 15 duplicates row 16 (`...3233133`).
- Before giving up it cycled through hypotheses about the 4s moving "toward" something (54 uses of "toward"). It never found the rule, under which each 4 moves to the side indicated by its enclosing shape's notch.

**28a6681f p0** (right shape, 6/100 wrong, 59k tok, Flash 3/3)
- The closest attempt to a solve. The concept is right: at @24k it says "Maybe the 1s are like 'water' or 'sand' that flows according to terrain", and it noted that the number of 1s is conserved. It correctly removes the top 1s.
- It never derived the precise fill rule (fill the basin between walls from the bottom up). Its final fill locations are a guess: "(9,8), (8,3), (8,8), (7,2), (7,3) ... This is a guess."
- It checked the guess against Ex2, found a mismatch ("Not matching"), and submitted the guess anyway.

## Patterns

**Counts (out of 10):**
- **Correct rule stated at any point: 0/10.** "Rule right, execution wrong": **0/10**.
- Partial rule (one sub-step right, the key step missing or wrong): **5/10**. These are b6f77b65, dd6b8c4b, both 247ef758, and 28a6681f.
- No rule found:
  - **4/10** gave a fallback answer: identity in 9bbf930d, an attempted identity in 88e364bc, all zeros in d35bdbdc p2, and a placeholder in 7c66cb00.
  - **1/10** hit the length cap in a degenerate loop (d35bdbdc p0).
- **Explicit surrender ("give up", "surrender", "placeholder", "output the test input"): 5/10.**
- **Submitted a rule it had already refuted on the training pairs: 4/10** (b6f77b65, 247ef758 p1, 28a6681f, dd6b8c4b).
- **Self-imposed time pressure: 9/9 finished traces.** "Given the time" or "out of time" appears 5–13 times per trace, first at 49–78% of the trace. Yet every finished attempt stopped at 58–95k tokens, under half the 200k budget. The model invents a deadline and quits early.
- **Perception and transcription errors: 6/10.** 7c66cb00 had the wrong grid size (25x25 instead of 30x24). 9bbf930d had column-count trouble. 247ef758 p1 had a wrong row length. 88e364bc corrupted the input while copying it. d35bdbdc p0 mixed row-index conventions. 247ef758 p0 used the wrong erase column.
- **The high cell accuracy is misleading.** The 97.5% (9bbf930d) and 96% (88e364bc) scores come from identity copies. The 61% for d35bdbdc p2 comes from an all-zero grid. The only real near-misses are 28a6681f (6 cells) and dd6b8c4b (12 cells), and both rest on partial rules.
- **Where the tokens went:**
  - Grid transcription: about 3–27% of characters (higher on the large grids).
  - Hypothesis cycling: most of the rest, with 100–270 "maybe / what if / perhaps" per trace and 11–21 explicit tests.
  - Verification against training pairs did happen, and individual checks were usually arithmetically accurate. But they ran on misread grids, and failed checks did not stop a submission.

**Verdict:**
1. None of these 10 is a "rule right, execution wrong" near-miss. The bottleneck is rule induction, along with misperceiving grid geometry. Faithful execution was not the problem, so a verification prompt has little to verify against.
2. Code execution could plausibly fix the perception errors (grid size, column counts, copying) and catch the 4/10 "submitted a refuted rule" cases. Still, at most 28a6681f and perhaps dd6b8c4b look convertible, because the model never states a correct transform for the rest.
3. More budget would not help as things stand: the model quits at under half the budget out of invented time pressure, or loops. Raising the cap does nothing unless the "Given the time" early-exit habit is suppressed, for example by telling it how much budget remains or by forcing a continuation.
