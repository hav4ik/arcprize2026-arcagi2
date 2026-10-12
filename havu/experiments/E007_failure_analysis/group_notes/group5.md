# Group 5: failure analysis (Flash, max effort, plain prompt)

There are 13 wrong attempts across 9 tasks. One attempt each on 7b3084d4 and a32d8b75 p0 was an api_error and is not counted.

| Category | n | Attempts |
|---|---|---|
| partial_rule | 8 | 2b83f449 a0, a1; a32d8b75 p0a1, p1a1; dfadab01 a0, a1; 35ab12c3 a0; b6f77b65 p0a0 |
| execution | 3 | 7b3084d4 a1; 581f7754 p1a0; 8b9c3697 a0 |
| perception | 1 | 16de56c4 p0a1 |
| ambiguous | 1 | 581f7754 p0a1 |

There were no pure missed_idea cases. Every wrong attempt had the main concept right.

## Per-task notes

- **2b83f449.** Both attempts got the 868 bars exactly right. Every error is in 3-ball placement.
  - a0 keeps the per-row 3 totals right but never puts a 3 beside a 0-hole. It moves that ball to the outer edge as `33`. This is a per-row formula with no per-region logic. Train 0 refutes it.
  - a1 drops 4 balls (14 vs 18).
  - This matches the earlier trace-based finding in E003 (part 2): a count formula, not conservation per component.
- **7b3084d4.** Assembly concept right; the top half (7 and 8 pieces) is exact. In the bottom half the 2 and 3 pieces sit in each other's slots.
  - The 2 piece is a valid 90° rotation.
  - The 3 piece is redrawn with 1-cell tabs that it does not have, though colour counts are preserved.
  - This is a hand piece-fitting failure. A brute-force fit in Python would find the unique assembly.
- **a32d8b75.** Attempt 1 is wrong on both pairs in the same way: the stamped mask is the truth rotated 180°, with correct tiles and colours. This is a systematic error in reading the orientation from the key box. Attempt 0 solved pair 1. I did not fully re-derive the key convention.
- **dfadab01.** Both attempts miss the exception that a marker touching a pre-drawn shape's corner draws nothing. Trains 0 and 2 show this exception. a1 also failed to clip the bottom-row stamps and output 21 rows. Its rows 0–19 are identical to a0.
- **16de56c4.** The pred is exactly the correct rule applied with the lone 4 read at column 8 instead of 7. That one misread flips the lattice check. The analogous case in row 5 was handled correctly.
- **35ab12c3.** The model joined the three lone 8 dots into an outline copied from the 3 shape. The truth leaves them alone, and train 1's three lone 3s refute the model's rule. Attempt 1 solved it.
- **581f7754.**
  - p0a1 left single-cell 2 markers in place. The train pairs have no singleton marker objects, so this is arguably ambiguous, but the natural generalisation moves them.
  - p1a0 moved an object correctly but shifted its stub one column to centre it (2 cells).
- **8b9c3697.** Rule and selection are right. The 0 trail overshoots by 2 cells.
- **b6f77b65.** The pred exactly matches "delete the 7 piece, then cell-wise column gravity" (verified in code). The truth uses rigid pieces with support, so nothing moves. This rule fails all 5 train pairs. Both attempts solved pair 1, so the gravity model was chosen inconsistently between calls.

## Cross-task patterns

1. **Missing sub-rules dominate (8/13), and 6 of those 8 fail the train pairs.** Those six are 2b83 a0, a32d ×2, dfad ×2, 35ab and b6f7 (the last also checks out in code). A `transform()` verified against the train pairs would have flagged them. Most are dropped exceptions:
   - the cancelled marker (dfad)
   - the unchanged lone dot set (35ab)
   - 3s beside holes (2b83)
2. **Pure machinery errors are rare but very cheap to fix.** 581f p1, 8b9c and 16de are 2–12-cell slips: one cell shifted, a trail too long, an off-by-one column read. 7b30 is the one large machinery failure (spatial fitting). Python execution would fix all four. A train-pair check would not catch them.
3. **Near misses are common.** 7 of the 13 have 12 or fewer wrong cells. The sibling attempt was correct in 6 of the 9 pairs that had a wrong attempt, so sampling plus a verifier would help.
4. **Same idea, different sub-rule between calls.** In a32d, b6f7 and 581f, the model applied a different convention than in its correct sibling call (orientation, gravity type, singleton handling). The errors look like the model fixed a sub-rule too early, not that it could not find the right one.
5. **Invariant checks would catch several.** These include colour and piece counts (2b83 a1 breaks 3-conservation; 7b30's 3-piece shape changes) and output size (dfad a1).
