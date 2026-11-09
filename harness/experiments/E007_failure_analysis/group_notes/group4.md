# Group 4: DeepSeek V4.1-Flash wrong attempts (idea vs machinery)

There are 16 wrong attempts across 10 tasks. Category counts:

- partial_rule: 9
- execution: 4
- missed_idea: 3
- perception: 0
- ambiguous: 0

Sibling outcome:

- 6 pairs had both attempts wrong: 269e22fb p0, 78332cb0 p0, 9bbf930d, de809cff.
- 8 wrong attempts had a correct sibling.

## Per task

- **269e22fb** (3 wrong; missed_idea ×3). Every input is a crop of one fixed 20x20 master picture, and the output is the full master in the matching orientation and colours.
  - p0a0 copied the crop correctly into the top-left and invented the rest, at about chance accuracy.
  - p0a1 tiled the input 2x2.
  - p1a0 made a 4-fold mirror-symmetric grid.
  - p1a1 recalled the master exactly. So recalling the template is possible, but the model rarely gets there.
- **78332cb0** (3 wrong; partial_rule ×3). Panels should be chained by matching edge contact points, and the chain sets both the order and the strip orientation. The model instead copied orientation rules from the train pairs: "2x2 -> vertical" (train 0) and "vertical -> horizontal reversed" (train 1). These shortcuts fit all the train pairs, so a code check would not catch them.
- **9bbf930d** (2 wrong; partial_rule, low confidence). The rule (same-colour-sandwiched corridor: flip the left 6 to 7, put a 6 at the open end) was nearly right: 11 of 13 changes are correct. Both attempts put the 6 at the wrong end of the corridor on rows 14 and 21, where the corridors cross vertical lines. The error is the same in both attempts, so it is systematic. Attempt 0 also has one copy slip.
- **de809cff** (2 wrong). Rule: holes become 8 with a 3x3 ring in the other region's colour, and noise is removed.
  - a0 is execution: one hole missed, one ring shifted by one column, one copy slip, 17 cells in a 30x30 grid.
  - a1 is partial_rule: it skipped holes on the region boundary, which train 0 does show.
- **16b78196** (1 wrong; partial_rule). Jigsaw pieces into wall notches. The wrong attempt put each piece in its own notch and missed the piece-on-piece stacking. Sibling correct.
- **332f06d7** (1 wrong; execution, low confidence). Moving the block through the maze: the model stopped it at a corridor turn on the direct route toward the 2. A 4-wide passage continues upward from there. This is a hand path-tracing or fit-check failure. Sibling correct.
- **4e34c42c** (1 wrong; execution, low confidence). 2D jigsaw assembly. All pieces are intact, but the joins and global layout are wrong (13x21 instead of 14x19). Sibling correct.
- **7ed72f31** (1 wrong; execution, high confidence). The reflections are perfect, but one row was inserted, giving 29x28 instead of 28x28. This is pure machinery.
- **b5ca7ac4** (1 wrong; partial_rule). Boxes slide to the wall by border colour. The model stacked them in the wrong order (the farther box ends up at the wall) and ignored a 1-row overlap. Running it on train 0 would expose this. Sibling correct.
- **fc7cae8d** (1 wrong; partial_rule, high confidence). The crop is right, but the model used the majority train orientation (rot270) and missed the flip that the test needs. The pred is the exact vertical mirror of the truth. Sibling correct.

## Cross-task patterns

1. **Most failures are rule-level, not arithmetic.** 12 of 16 wrong attempts are missed_idea or partial_rule. Only 2 are clean machinery errors that a Python tool would surely fix: de809cff a0 and 7ed72f31 p1a1. Two more (332f06d7, 4e34c42c) look like search or path-tracing failures; code might fix them, but that is uncertain.
2. **Copying surface patterns from train.** In 78332cb0 and fc7cae8d the model copied the orientation or order seen in train pairs instead of the underlying condition. The shortcut fits the train pairs, so self-checking on train with code would not help. Only rule induction would.
3. **High variance between the two attempts.** In 8 of 16 wrong attempts the sibling was correct: 269e22fb p1, 78332cb0 p1, 16b78196, 332f06d7, 4e34c42c, 7ed72f31, b5ca7ac4, fc7cae8d. The model often has the idea in one sample and not the other, so more samples plus a train-pair verifier would recover several of these. That works where the wrong variant fails train (16b78196, b5ca7ac4, fc7cae8d, de809cff a1), but not for the 78332cb0 shortcuts.
4. **Missing sub-conditions.** In 9bbf930d, de809cff a1, b5ca7ac4 and fc7cae8d the model left out a condition the train pairs do contain: boundary holes, overlap ordering, the flip case, and corridor ends. These are the cases where "write transform(), run it on train" would most likely catch the error.
5. **Template or memory tasks (269e22fb).** The model falls back to symmetry or tiling unless it notices that all train outputs are the same picture.
