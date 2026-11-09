# Group 0: failure analysis (DeepSeek V4.1-Flash, max effort, plain prompt)

Tasks: 13e47133 4c7dc4dd 8698868d abc82100 e87109e9 221dfab4 3dc255db 64efde09 a251c730 cb2d8a2c

20 wrong attempts: **execution 8, missed_idea 6, partial_rule 6** (no perception or ambiguous cases).
Of these, 12 would be caught by running a transform on train (all of them, really), and 8 are pure machinery errors that a tool would fix.

## Per task

- **13e47133** (both pairs, both attempts wrong; 4/4 execution). Rule: rings at Chebyshev distance from the region boundary, with colours cycling through the corner-diagonal markers. I checked this against both test outputs: it reproduces them exactly. All 4 preds get every rectangular region right and fail only where rings bend around concave corners of L-shaped regions. Pair 1 attempt 0 also copied a wrong colour (9 for 3) into one ring. This is a textbook machinery failure: a 10-line BFS would solve it.
- **4c7dc4dd** (4/4 missed_idea). The output sizes are right, but the content comes from the wrong box: it copied the 0/4/2 plus box instead of recolouring the 8/1 lattice box. I did not fully derive the true rule.
- **8698868d** (attempt 1 partial_rule; attempt 0 was an api_error). Compositing and rendering are perfect, but the shape-to-panel matching ignores "holes == dots" (4 of 6 assignments wrong).
- **abc82100** (2/2 missed_idea). The legend/stamp concept is absent: the preds are stripe textures or stamps piled in the last columns.
- **e87109e9** (2/2 partial_rule). Attempt 1 is almost right: its beams stop when they meet another beam, although train 2 shows beams crossing. Attempt 0 has wrong turn directions for several colours.
- **221dfab4** (one wrong attempt per pair; the sibling is correct each time). Pair 0: a dropped last row plus the 4,4,3 phase shifted by 2 (execution). Pair 1: band cells at odd offsets were not reset to background (partial_rule).
- **3dc255db** (sibling correct). The emitted line lengths are miscounted (2 instead of 4, 1 instead of 2) and the 2-line exits one row off (execution).
- **64efde09** (sibling correct). The 9 and 7 shadows are correct, but colour 6 was given the wrong role: lines from the caps instead of the middle shadow (partial_rule).
- **a251c730** (sibling correct). An extra interior row gives 10x14 instead of 9x14 (execution, off-by-one height).
- **cb2d8a2c** (siblings correct). Pair 1: the last row was dropped, and everything else matches (execution). Pair 0: the 3-path hugs wall ends with zero clearance (partial_rule).

## Cross-task patterns

1. **Single-attempt failures on solvable tasks are mostly machinery.** In 221dfab4 p0, 3dc255db, a251c730 and cb2d8a2c p1 the sibling attempt was correct and the wrong one is off by a dropped or added row, a miscount, or a phase shift. In these cases 2-attempt pass@2 already hides the problem, but pass@1 suffers.
2. **Dropped or added rows on large grids** (221dfab4, a251c730, cb2d8a2c p1): the output height is off by one while the content is otherwise identical. Just checking the output shape against the expected size would catch these.
3. **Geometric fills of non-convex regions** (13e47133) fail by hand even when the rule is clearly understood. These are the strongest candidates for a code tool.
4. **Partial rules usually drop one sub-rule that a train pair does exhibit** (e87109e9 beam crossing, 221dfab4 odd-offset reset, cb2d8a2c clearance, 64efde09 one colour's role). Verifying a written transform on train would flag them, though not necessarily fix them.
5. **Genuine concept misses** (4c7dc4dd, abc82100) produce outputs with the right size but unrelated content, and they cost about 95k tokens per attempt.
