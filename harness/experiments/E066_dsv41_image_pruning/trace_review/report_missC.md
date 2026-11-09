# Failure review, set C (candidate 1 vs E72-152 vs full model)

Traces: `missC/{c1,e72,ref}/<task>_a<k>.txt` (ref = self-hosted full model). Wrong cells were computed by comparing each
record's `pred` with the ground truth in `data/arc-agi_evaluation_solutions.json`. Observation is marked OBS, inference INF.
Not finished or missing (no trace): c1 on 7b80bb43, 800d221b; e72 on 3dc255db.

## Table

| task | model | attempt | cause | same failure as full model? | pruning-related? |
|---|---|---|---|---|---|
| 3dc255db | c1 | a0 (wrong, 1 cell) | Ad-hoc hand-coded test edit (no general `solve`): the test-specific removal list included a red 2 at (9,7) that is not a marker, so one cell was erased. Direction was chosen by a centroid heuristic; training was never run through code. | no (ref both right; ref's rule used the same idea but kept that cell) | unclear (looks like sampling variance; a1 used a bbox filter and was right) |
| 3dc255db | c1 | a1 (right) | Same rule, marker cells selected by bbox filter; first wrote `answer=b.copy()` with a stale variable, noticed, recomputed. | n/a | n/a |
| e3721c99 | c1 | a1 (test 1 wrong, 120 cells) | Hole counting used 8-connected background, so diagonally touching holes merged (counts 3->2, 4->3 on two grey blobs, shifting them to the wrong legend colour). Training passed under both connectivities, so nothing forced a choice; never tested the 4-conn alternative on test. | no (ref used 4-conn; right) | unclear (a0 of c1 used the default 4-conn and was right; e72 a1 explicitly noticed the 4/8 ambiguity and resolved it) |
| 7b5033c1 | c1 | a0 | Training-consistent shortcut "count per colour, order by first appearance" with a typo (`mmost_common`, fixed). Test-only feature missed: the shape is a path where colours repeat (1,3,...,4,1,3). | yes (ref a0 and a1 gave the identical wrong 25x1 answer) | no |
| 7b5033c1 | e72 | a0 | Shortcut "flatten non-background cells row-major"; passes both training pairs, wrong on test (6 cells). Stopped after 2 turns / 764 tokens. | partly (different shortcut, same shallow family) | unclear |
| 7b5033c1 | c1 / e72 | a1 (right) | c1 a1 found each non-bg set is a simple path and walked it from the top endpoint; e72 a1 also right. | no (ref a1 wrong) | no |
| dbff022c | c1 | a0, a1 | Wrong rule: legend pairs read as top->bottom only (source colour -> fill colour), so only the grey shape was filled. Truth also fills the blue and green shapes (legend used in both directions) and does not fill (6,8..10). 18 wrong cells. | yes (all 6 attempts, c1/e72/ref, are cell-identical) | no |
| dbff022c | e72 | a0, a1 | Same wrong prediction | yes | no |
| 88e364bc | c1 | a1 (test 0 wrong, 2 cells) | Slide-to-wall rule for the 4s: one 4 slid to (8,16) but truth keeps it at (4,12). Training passes. | yes (all 5 attempts have the same two wrong cells) | no |
| 88e364bc | e72 | a0, a1 | Same two cells | yes | no |
| 7b80bb43 | e72 | a1 (2 cells) | Gap-filling heuristic erased diagonal noise cells and filled gaps; residual errors (15,2),(16,3). | partly (ref has 1 wrong cell, (15,2), and the same code idea) | no |
| 800d221b | e72 | a1 (5 cells) | Block-ring colouring propagation; mis-coloured cells near (11,15..18),(12,13). | partly (ref misses 3 or 5 cells in overlapping positions, but not the same ones) | no |

## Observations
- OBS: on dbff022c and 88e364bc every attempt of c1, e72 and ref (5-6 attempts each) produced the same wrong grid. The models agree with each other and with the full model on the wrong hypothesis. ref a1 on dbff022c wrote "Only 5 is a source ... That seems like a lot unchanged", checked it against the training orientation and kept it (ref/dbff022c_a1.txt).
- OBS: the two cases where c1 differs from its own good attempt or the full model (3dc255db a0, e3721c99 a1) are small, local decisions (a hand-picked marker cell list; 8-conn vs 4-conn), not misreads of the grid or tool-format failures. No degenerate text, no typo-driven failure beyond one fixed `mmost_common`, no budget exhaustion (all finished by `submit`, 3-17 turns, 1-21k tokens).
- OBS: 7b5033c1 is the shallowest set: ref both attempts took 3 turns / ~500 tokens; c1 a0 3 turns, e72 a0 2 turns. The attempts that were right (c1 a1 10 turns, e72 a1 6 turns) explored the structure. ref was wrong on both.
- OBS, c1 vs e72: both submitted after the first training-passing program; depth and length were similar (c1 1-21k tokens; e72 0.8-23k). e72 showed more explicit ambiguity checking on e3721c99 (4/8-conn, listing both counts) and was right on both attempts there; c1 has the one wrong. With n=1-2 attempts per case this is not a measurable difference.
- INF: the shared misses are task difficulty (hypothesis that fits all training pairs but not the test), not pruning damage. The candidate-only misses (3dc255db a0, e3721c99 a1) look like sampling variance at the level of the full model's second attempts, with at most a weak sign of less test-time checking in the pruned models.

## Quotes
- c1/7b5033c1_a0.txt: "Counts: 1 eight, 3 five, 8 six, 4 six. Need submit." (the earlier reasoning had noted "Wait color 1 in later area too" for the second cluster of cells, but the first-appearance ordering was never questioned).
- ref/dbff022c_a1.txt: "Hmm but then blue and green and red remain unchanged. That seems like a lot unchanged." followed by "We should rely on fixed orientation from ex0 ... Solid."
- e72/e3721c99_a1.txt: "comp7 holes4=4, holes8=3. If samples only holes1,2,3, so comp7 holes4->0 or holes3->2. This ambiguity affects output visibly." (c1 a1 never raised it).
