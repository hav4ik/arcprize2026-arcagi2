# Group 6: failure analysis (DeepSeek V4.1-Flash, max effort, plain prompt)

Tasks: 446ef5d2 7b80bb43 aa4ec2a5 e12f9a14 1ae2feb7 36a08778 62593bfd 9aaea919 c7f57c3e
There were 14 wrong attempts. By category: execution 7, partial_rule 6, other 1. No missed_idea or perception.

## Per task

- **446ef5d2** (jigsaw: assemble rectangle fragments into stacked sub-frames, anchored at an L-marker). Pair 1 (20x20, 2 sub-frames) was solved by both attempts. Pair 0 (26x26, 3 sub-frames) failed both:
  - a0 got the anchor corner and total area (220) right. It then filled a transposed 10x22 box with an incoherent packing, and the colours were not conserved (3: 4 cells vs 22). Classed execution: the search failed.
  - a1 returned the input almost unchanged (it only recoloured the marker) after 131k tokens. Classed other: it gave up.
- **7b80bb43** (bridge gaps that have a bent-off stray attached; erase other strays). Both attempts were within 2–3 cells of the truth.
  - a0 also bridged a gap whose stray is detached. Train 0 shows this case is not bridged.
  - a1 "straightened" a stray hanging off an unbroken line instead of erasing it. This case does not occur in train.
  - Both classed partial_rule, at medium and low confidence.
- **aa4ec2a5**: the content is exactly right in both attempts. a0 adds an extra row (27 rows) and a1 drops the last row (25 rows). Pure execution.
- **e12f9a14** (rays through frame gaps, deflecting alongside other rays). The rule, including the subtle deflection, was understood.
  - Pair 0: the model read a frame gap one column off, which flipped a ray's direction, and shifted the bottom object down one row.
  - Pair 1: one cell is missing at a deflection on the grid edge.
  - The other attempt on each pair was an api_error.
- **1ae2feb7**: rule right. The output has 9 rows instead of 10 (a wall row was dropped). Execution.
- **36a08778**: water flow is right. One input red bar was copied one column to the right (10 cells). Execution.
- **62593bfd** (objects go to the top or bottom edge). The true criterion is "cell count shared with another object → top; unique → bottom". Both attempt-0s used a wrong criterion: 2 objects were swapped in pair 0 and 3 in pair 1, and pair 1 also lost a row. Classed partial_rule. Attempt 1 was correct on both pairs.
- **9aaea919**: the model found both operations (stack copies, recolour a column) but bound them to the wrong bar colours and recoloured to 7 instead of 5. Classed partial_rule. The sibling attempt was correct.
- **c7f57c3e**: the centre colours were swapped correctly, but the caps were recoloured in place instead of moving to the opposite side. Train 1 contradicts that. Classed partial_rule. The sibling attempt was correct.

## Cross-task patterns

1. **Grid-size and row-bookkeeping slips are common, even when the content is perfect:**
   - aa4ec2a5 (+1 row in one attempt, −1 row in the other)
   - 1ae2feb7 (−1 row)
   - 62593bfd p1 (−1 row)
   - e12f9a14 p0 (object shifted one row)

   A trivial "output shape == input shape" check, or writing the output with code, would fix 3 of these outright.
2. **Off-by-one copying of input objects in large (≥26x26) grids:**
   - 36a08778: a bar shifted one column
   - e12f9a14: a gap misread one column off, and an object shifted one row

   The rule is right but the coordinates drift. Parsing the input in code would remove this.
3. **Partial rules are mostly the "selection/binding" part, not the mechanism:**
   - 62593bfd: which objects go up
   - 9aaea919: which bar triggers which operation
   - c7f57c3e: whether caps move
   - 7b80bb43: when a stray counts as attached

   In 4 of 6 partial_rule cases the sibling attempt got it right, so sampling variance is high. A train-pair code check would have flagged most of them, since the wrong criterion contradicts a train pair.
4. **Large-instance collapse** (446ef5d2): the same concept was solved on the small test pair but failed on the larger one. The model either produced a non-conserving fake assembly or returned the input unchanged. This looks like a search or budget limit rather than a missing idea.
5. **Near misses:** 7 of the 14 wrong attempts have 10 or fewer wrong cells or a pure shape mismatch. For these, "wrong" overstates how far the model was from solving the task.
