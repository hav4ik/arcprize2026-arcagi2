# Group 2 failure analysis (DeepSeek V4.1-Flash, max effort, plain prompt)

17 wrong attempts across 10 tasks (e3721c99 pair 0: both api_error, not counted).

| category | n |
|---|---|
| partial_rule | 8 |
| execution | 5 |
| missed_idea | 2 |
| perception | 1 |
| ambiguous | 1 |

## Per task
- **21897d95**: Regions get recoloured according to arrow markers, and some outputs are rotated 90 deg. P0a1 only erased the markers and copied the input (missed_idea; its sibling solved the pair). P1a0 has exactly the right colours but is not rotated: pred == rot90(true,-1). The square train pairs are not rotated either, so this is ambiguous. P1a1 has an incoherent region layout (missed_idea).
- **6e4f6532**: Both wrong attempts have the right idea and fail on copying. a0 has one wrong cell (anchor arm (20,5) drawn as 7). a1 dropped a blank row, so the output is 29 rows instead of 30.
- **88e364bc**: Both attempts are identical with 2 wrong cells. A 4 moving diagonally slipped past a wall it was already touching orthogonally, ending at (8,16) instead of (4,12). The stop condition is wrong (partial_rule). The train pairs may not separate the two readings.
- **d35bdbdc**: Both attempts are identical. The pointer pairing is right, but in the chain 4->2->3 the model kept the wrong shape. It ignored the rule that only shapes at the ends of the 5-path survive (partial_rule).
- **0934a4d8**: Symmetry fill. 6 cells in the transpose-only corner were read from the wrong source cells (execution).
- **2d0172a1**: a1(p0) drew the inner box with one dot instead of two. Its output is exactly the train-3 template recoloured, so it lost an object, or snapped to a memorised template (perception). a1(p1) dropped the trailing padding row (execution).
- **4c3d4a41**: Uses a rigid upward shift by h where the true rule absorbs empty cells first. Train 0 col 11 contradicts the rigid shift (partial_rule).
- **71e489b6**: Both wrong attempts remove isolated stray 1-pixels but keep single-pixel 1-bumps attached to a region. Train 2 (3,6) shows that such bumps are removed (partial_rule).
- **a47bf94d**: The concept is right, but only 2 of 5 links through the 8-line network were traced. The 3 longest links were dropped (execution: path tracing).
- **e3721c99**: Blobs with 0 holes were painted 4, because 4 was the solid key in train 0. In this test legend 4 means 1 hole, and blobs with an unmatched hole count should be deleted. The model matched by colour, not hole count. There is also a 5-cell legend copy error (partial_rule).

## Cross-task patterns
1. **The idea is usually there.** In 8 of 10 tasks the sibling attempt or another test pair was solved. Most failures are close misses (1-14 wrong cells): 6e4f6532, 88e364bc, 71e489b6, 0934a4d8, 4c3d4a41, a47bf94d.
2. **Partial rules are the biggest bucket, and most are cases that only appear in the test.** Examples: a diagonal stop condition (88e364bc), noise attached to an edge (71e489b6), gap absorption (4c3d4a41), a 0-hole blob with no legend match (e3721c99), choosing survivors in a chain (d35bdbdc). In about 4 of 8, a train case contradicts the model's rule (4c3d4a41 train0, 71e489b6 train2, e3721c99 train1, d35bdbdc train1). Running a `transform()` against the train pairs would have flagged these.
3. **Pure execution errors are frequent and cheap to fix with a tool.** Examples: dropped rows or cells (6e4f6532 x2, 2d0172a1 p1), wrong symmetry lookup (0934a4d8), incomplete path tracing (a47bf94d). Python execution would likely fix all 5.
4. **The model copies values from train outputs.** e3721c99 reused "4 = solid" from train 0, and 2d0172a1 p0 output exactly the train-3 schematic.
5. **Both attempts often fail the same way:** 88e364bc and d35bdbdc give identical preds twice. Sampling a second attempt adds little diversity when the error is a rule gap.
