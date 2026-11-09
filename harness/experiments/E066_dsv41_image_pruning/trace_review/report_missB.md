# Miss review B: E72-152 misses on tasks the full model solves

Traces: `missB/e72` (dsv41_low_e066_e72_img), `missB/ref` (full self-hosted), `missB/ref_api` (API pilot30). `missB/c1` is empty: candidate 1 (c1x_img) has no finished attempt on these tasks. The E72-152 run has no finished attempt for 247ef758 a0 and 6ffbe589 a0, so only the attempts listed below exist.

## Table

| task | att | cause | divergence point (one line) | pruning-related |
|---|---|---|---|---|
| 35ab12c3 | a0 | wrong hypothesis fitting train, test-only feature | Chose "convex hull of each colour's dots, draw aligned hull edges"; ref chose "connect same-colour dots in angular order along aligned (H/V/diagonal) segments". They differ only on the test's non-convex colour 3 (interior dots (6,10),(6,13)); E72's output lacks row-6 cols 10-13 and the (6,13)-(8,15) diagonal. | no (plausible variance) |
| 35ab12c3 | a1 | same | Same hull rule; wrote "interior points ... lie inside? ... likely" and moved on without testing a non-convex alternative. 3 tool errors (a lost sandbox state, a qhull error and a numpy shape bug) were fixed normally. | no / unclear (shallow test check) |
| 71e489b6 | a0 | ad hoc rule fit to train, over-generalised | Rule: a 0 cell is a "seed" if its zero-neighbour count is <= 1 (or in a small component), 3x3 orange halo; blue cells with <= 1 neighbour erased. Ref/API rule: black holes inside blue get halos, blue noise inside black is erased, using the cleaned grid. E72 therefore adds halos at "background tips" in the test (e.g. (3,11), (5,15), (11,6)). | unclear (weak abstraction; the full model needed 47k tokens and also gets 1/2 pairs self-hosted) |
| 71e489b6 | a1 | same | Same "seed = n4 <= 1 or component size <= 3" patch rule (`solve2`); wrote "Since training is exact, submit" despite noting the output "weird". | unclear |
| 6ffbe589 | a1 | missed test-only feature | Rule (rotate each colour's layer by the number of stray cells outside the tile) is right. Tile found as the "largest 8-connected component" bbox = 7x7 green shape. The test's tile is 13x13 and includes the separated grey 5 pieces (ref bbox rows 6-19, cols 7-20). E72 treated the 5s as stray noise and returned 7x7. | no (the full model also fails a1 of this task; ref a0 right) |
| 247ef758 | a1 | 50/50 guess on layering | Test 1: shapes 4 and 6 overlap at (6,12). Truth has 6 on top; E72 decided "reference ... 4 on top" ("Without knowing, 50/50") from a smaller/compact on top heuristic. Test 0 correct. | no (guess; ref a0 is also partial) |

## Tokens and turns (E72 vs full)

| task | E72 turns / out tokens | full self-hosted | full API |
|---|---|---|---|
| 35ab12c3 | a0 12 / 31.1k, a1 13 / 25.0k | 7 / 24.1k, 18 / 26.9k | 7 / 22.0k, 8 / 17.5k |
| 71e489b6 | a0 16 / 23.2k, a1 13 / 23.1k | 14 / 32.7k, 17 / 28.0k | 19 / 47.3k, 22 / 32.6k |
| 6ffbe589 | a1 15 / 24.7k | a0 18 / 19.8k, a1 39 / 34.5k | a0 18 / 19.0k, a1 21 / 40.3k |
| 247ef758 | a1 10 / 28.9k | a0 23 / 30.6k, a1 14 / 28.0k | a0 15 / 21.1k, a1 9 / 31.1k |

No budget or turn exhaustion: all six attempts ended `submitted` with 23-31k tokens, inside the 40k soft budget.

## Observations
- All six attempts submit through the normal flow, all pass every training pair with their program , and none shows garbled tokens, broken tool calls, rescues, or misread grid values.
- Tool errors are ordinary coding mistakes (35ab12c3 a1: qhull on collinear points, a numpy broadcast bug, one lost sandbox state; 247ef758 a1: a dict passed to np.array), each fixed in the next turn.
- Every miss is a rule that fits the training pairs but generalises differently on a test-only situation (non-convex colour set; tips touching the background; a tile with detached pieces; an overlap not in training).
- E72's attempts are shorter in turns on 71e489b6 and 6ffbe589 than the full model's, and reach a rule sooner.

## Inference
- The pattern is "commit to the first training-consistent rule and submit", with little probing of the test input for ways the rule could differ. The full model's successful traces found the more natural rule (angular connection, hole-in-blue semantics, tile = border-bounded square). That could be weaker abstraction from pruning, but the full model also fails 6ffbe589 a1 and parts of 71e489b6 and 247ef758, so variance explains at least those.
- Pruning-related verdict: no mechanical damage; possible shallower exploration, unproven. 35ab12c3 (both attempts the same wrong hypothesis) and 71e489b6 (both the same patch rule) are the strongest hints of a systematic bias, not of broken mechanics.

## Quotes
- `missB/e72/35ab12c3_a1.txt`: "The interior points (6,10),(6,13) lie inside? Original c3 at (6,10),(6,13) are included as original points, output includes them. Are they hull vertices? ... likely"
- `missB/e72/71e489b6_a1.txt`: "Our output has weird rings around many seeds and blue removals." ... "Since all training perfect, submit."
- `missB/e72/247ef758_a1.txt`: "Which looks more plausible? Without knowing, 50/50."
