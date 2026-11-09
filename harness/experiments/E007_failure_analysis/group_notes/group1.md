# Group 1 failure analysis (DeepSeek V4.1-Flash, thinking max, plain prompt)

19 wrong attempts across 10 tasks. Category counts: partial_rule 8, missed_idea 5, execution 5, ambiguous 1.
Rows are in `group1.jsonl`. No traces were available, so the model's rule is inferred from its output (5545f144 also draws on E003 part 2).

## Per task

- **195c6913** (4 wrong, both pairs 0/2). The rule is a sequence path from the left edge, with straight runs that turn at obstacles and a marker colour on the blocked cell.
  - p0a1 (4 cells) and p1a1 (3 cells) are near misses. One overshoots by one cell, one keeps turning at the grid edge instead of stopping, and there are stray background copy slips. These are pure machinery errors on 30x30 grids.
  - p0a0 traced the region boundary instead (missed_idea, 186 cells).
  - p1a0 dropped the turn-marker colour and made routing errors (partial_rule, low confidence).
- **5545f144** (2 wrong, 0/2). Both attempts are identical: keep the shape, delete the dots. This is the same give-up fallback the E003 traces showed. missed_idea.
- **88bcf3b4** (4 wrong, 0/2 on both pairs). Snake re-wrapping around a target.
  - p1a1 is pure execution: two single cells misplaced, snake length conserved.
  - p0a0 and p1a0 got one or two systems exactly right but used the wrong wrap geometry or tail direction on another. The true geometric rule is hard to state crisply, so these are partial_rule with low confidence.
  - p0a1 drew a frame around the target instead (missed_idea).
- **b9e38dc0** (2 wrong). Interior fill was right in both attempts.
  - a0 used a fixed 45-degree cone; the truth continues the wall slope, 1 row per 2 columns.
  - a1 drew no cone at all, and its output is 14 rows instead of 15.
  - Both are partial_rule, and train 2 would expose them.
- **faa9f03d** (2 wrong). Path-joining with 2 (corner) and 4 (continue to edge) markers.
  - a0 had the main loop right but kept dangling segments, got crossing priority wrong, and its output is 11 rows instead of 12.
  - a1 misread the markers (missed_idea).
- **271d71e2** (1 wrong, sibling correct). The big box was right. The small box slid 4 instead of 3 rows (its handle merged into the wall), although it converted only three 5s. This is an off-by-one in the shift distance (execution).
- **409aa875** (1 wrong, sibling correct). 2 cells wrong: one shape's projection was dropped, and a collision cell got 9 instead of 1. The 1-on-collision sub-rule is shown in train 0 (partial_rule).
- **6e453dd6** (1 wrong, sibling correct). Rows got 222 when the shape had any gap or stem, not only fully enclosed holes. Train 1 row 10 contradicts this (partial_rule; a code check would catch it).
- **a25697e4** (1 wrong, sibling correct). The left docking system is exact. On the right system the two colours are swapped, and the 2-block gained an extra row.
  - Classed as execution (inconsistent across objects, duplicated row), low confidence.
  - The full colour-exchange rule was not pinned down.
- **dbff022c** (1 wrong, sibling correct). The legend was read with the top row as the key, the opposite of the true direction, and applied consistently: only the 5-shape was filled. Every train legend sits on the top or left edge, so "top/left is the key" and "the grid-edge side is the key" are not distinguished (ambiguous).

## Cross-task patterns

1. **Near misses on big grids are machinery errors.** 195c6913 p0a1/p1a1, 88bcf3b4 p1a1, 271d71e2 and a25697e4 have 3–17 wrong cells out of 600–900. They include overshoots, single misplaced cells, background cells copied wrongly, and a duplicated row. A Python transform would fix these if the model could write the rule.
2. **Wrong output height (a dropped or duplicated row).** b9e38dc0 a1 has 14 rows instead of 15, faa9f03d a0 has 11 instead of 12, and a25697e4 has a 5-row block where the input has 4. Row bookkeeping on long grids is fragile.
3. **Sub-rule underfit that a train check would expose.** b9e38dc0 (cone slope; train 2), 6e453dd6 (enclosure; train 1), 409aa875 (collision gives 1; train 0) and 195c6913 p1a0 (marker colour). In each, the model had the core idea, but the dropped condition is visible in at least one train pair. Running the rule on the train pairs would flag these, but would not fix them.
4. **When a sibling attempt was correct (5 tasks), every wrong sibling was a small deviation.** These were partial_rule, execution or ambiguous, never missed_idea. Sampling variance in sub-rule choice and execution decides these tasks, not concept discovery.
5. **missed_idea appears where both attempts fail.** In 5545f144, 88bcf3b4 p0a1, 195c6913 p0a0 and faa9f03d a1, the model falls back to a simpler concept (identity minus dots, framing, contour tracing).
