# Review A: E72-152 misses where candidate 1 (or the full model) got the same slot right

Traces: `missA/{e72,c1,ref}/<task>_a<k>.txt`, index in `missA/*/index.tsv`.
Observation = what is in the traces. Inference = my reading. I re-ran the submitted programs of the two sides on the test inputs (from `data/arc-agi_evaluation_challenges.json`) to locate the exact difference. This is not ground truth, but the diff is exact.

## Summary table

| task | attempt | turns / out tokens (failing vs siblings) | cause | divergence point | pruning-related |
|---|---|---|---|---|---|
| 1ae2feb7 | e72 a0 (2/3) | 8 / 7.3k vs c1 7 / 11.3k, 9 / 11.1k; e72 a1 6 / 10.5k (3/3) | Test-specific feature unchecked | Rule fixed as "pattern left of divider -> right"; test 2 has the pattern on the right and an empty left side. It printed test 2 unchanged, called it "mostly input", and submitted. | no (sampling) |
| 31f7f899 | e72 a1 (0/1) | 4 / 3.2k vs e72 a0 9 / 16.0k (right), c1 5 / 9.7k and 7 / 6.8k, ref 4 / 6.4k | Coding bug: even height | `range(R-hh//2, R+hh//2+1)` draws 11 cells for height 10. Train heights are all odd. Test needs height 10. It never counted the rows. | unclear, leaning no (hasty, thin verification) |
| 36a08778 | e72 a1 (1/2) | 13 / 26.1k vs c1 12 / 20.0k and 8 / 10.5k | Coding bug: red overwritten | In the final `out`, top-row cells of a bar are written as 6 without `a!=2`. In test 1 this turns red (16,2) into 6. c1 had `if a[rr,cc]!=2`. One cell is wrong. | no |
| b0039139 | e72 a1 (0/2) | 8 / 9.5k vs c1 8 / 18.8k, e72 a0 5 / 11.5k (right) | Wrong hypothesis that fits train | Copy count k = height of the yellow shape. This equals the number of green components in all 4 train pairs. It never used the green region. Test0 needs 5 copies (k=Y_h gives 3), test1 needs 6 (Y_h gives 5). e72 a0 tested green component counts and was right. | no (ordinary sampling) |
| 13e47133 | e72 a1 (1/2) | 18 / 28.1k vs ref 14 / 21.2k | Wrong hypothesis that fits train, plus ignored warning | Colour sequence read along a diagonal from a bounding-box corner. Train components are rectangular, so it works. Test 1 region 5 (a notched region) has its seeds on a diagonal from a local corner, (16,8), (17,9), (18,10). The debug print showed `(5, ..., [8])` (no sequence, fill with background) and it moved on. Ref keys seeds by 8-neighbour peel depth. 336 cells differ in test 1 (rows 9-29). Test 0 is identical. | no / unclear |

## Details per case

### 1ae2feb7 (e72 a0 vs c1)
- Observation: all three solve the "period = run length, rightmost run wins" rule and pass all train pairs. c1 a0 explicitly writes in its reasoning "Then determine source side" and handles a source on the right, with the divider counted from the nearest column. e72 a0 only has `for p in range(0,W-sep-1,L): out[r,sep+1+p]=c`. Test 2 has colours at cols 13-14 and an empty left. e72 a0 printed that test output (unchanged) and wrote "right width only 2, so output mostly input".
- Observation: e72 a1 handled the mirror case (distance from the divider, painting far to near) and got 3/3.
- Inference: the miss is unchecked test-specific input geometry. The test-input grid was visible in the trace.

### 31f7f899 (e72 a1 vs c1, ref)
- Observation: the rule is the same everywhere: sort bar heights left to right, redraw centred on the magenta row. e72 a1 writes `for r in range(R-hh//2, R+hh//2+1)`. For hh=10 and R=8 this gives rows 3-13 (11 cells). c1 uses `start=axis-T//2; end=start+T-1` (rows 3-12) and the ref uses the original spans.
- Observation: e72 a1's test output shows 11 sevens in col 16. Its reasoning says "orange col16 height10" and never counts. 4 turns and 3.2k tokens is the shortest attempt of all (c1 6.8k to 9.7k; ref 4.8k to 6.4k). The same model's a0 (9 turns, 16k) checked more and got it right.
- Inference: unverified edge case (even height). Train has only odd heights, so the training check cannot catch it. A thinner verification effort, but the evidence for pruning is weak: ref also takes 4 turns.

### 36a08778 (e72 a1 vs c1)
- Observation: both use the same beam/frame rule. Diff of the two programs on test 1: exactly one cell, (16,2), 6 in e72 and 2 in c1. The e72 version adds top-row cells without skipping red, then `for p in out: b[p]=6`.
- Observation: the e72 trace worried about vertical red components ("Interesting: comp5 vertical...") for several turns and then submitted. It did not diff the output against the input for changed red cells.
- Inference: a coding bug in a 26k-token attempt; the rule and the verification effort are comparable to c1's. Ordinary variance.

### b0039139 (e72 a1 vs c1, e72 a0)
- Observation: e72 a1 at step 13 wrote "Output height seems..." and compared to G dims, but then fixed k=Y.shape[0] ("k = Y height"), reproduced 4/4 train and printed test shapes (17x3, 5x24). It never looked at G in the solver. c1 and e72 a0 count 8-connected green components (test0 = 5, test1 = 6) and give shapes 19x5 and 5x29.
- Observation: e72 a0 writes "Count maybe equals number of distinct segments in green... Components..." (explores before fitting).
- Inference: first hypothesis that fits all train pairs is accepted without testing the alternative. Sampling variance: the other attempt of the same model explores it.

### 13e47133 (e72 a1 vs ref a1; c1 unfinished)
- Observation: both use 8-neighbour erosion/peel distance with a colour cycle. The difference is the sequence: e72 reads it along a diagonal from a bbox corner; ref reads it at the depth of each seed. On test 1 e72 gives region 5 seq `[8]` (all background) where ref gives `[4, 9, 2]` (seeds at (16,8), (17,9), (18,10)); regions 1-4 and 6 agree. Test 0 matches ref exactly.
- Observation: e72's debug print at step 48 lists `(5, (9,29,8,29), 'none', [8], None)`; the next message comments only on test 0. Also one "Python state lost" error (infrastructure, same event seen in c1's b0039139).
- Inference: wrong hypothesis plus a visible warning not acted on.

## Pattern across the 5 cases
1. Observation: in 4 of 5 cases, the E72-152 programs were correct on every training pair and the failure is in a test-specific feature: mirrored geometry (1ae2feb7), an even bar height (31f7f899), an overlap of a frame with a vertical red bar (36a08778), a green-region count that differs from the height (b0039139), a notched region (13e47133). The sibling programs handle it, either by a more general rule or by one explicit line.
2. Observation: in every case the evidence was one print away. E72-152 printed the failing test output or the diagnostic (`'none', [8]`, unchanged grid, 11-cell column) and did not comment on it.
3. Observation: no garbled tokens, no format breakdown, no wrong grid reads and no budget exhaustion in any failing attempt (all stop=submitted, 0 rescues). Tool errors: 0 except one "state lost" in 13e47133 a1.
4. Observation: in 4 of 5 cases the same model's other attempt is correct on the same task (b0039139 a0, 31f7f899 a0, 36a08778 a0, 1ae2feb7 a1), with a longer or more exploratory trace.
5. Inference: "submits after training pairs pass without a test-specific check" fits all five and is a verification weakness rather than broken mechanics. It could be pruning-related (weaker self-checking) but the cases are equally explained by sampling variance. I see no evidence of mechanical damage in these five. The failing attempts are shorter in 31f7f899 and 1ae2feb7 but not in the others (36a08778 and 13e47133 are longer than the siblings).

## Quotes
- `missA/e72/1ae2feb7_a0.txt`: "Check test1/test2 right width only 2, so output mostly input with period fills."
- `missA/e72/31f7f899_a1.txt`: "I think answer = solve(test_input)... Wait test bars: ... Submit." (the tool call `submit_answer` follows `solve(test_input)` directly after printing an 11-cell orange column that it described as "height10").
- `missA/e72/b0039139_a1.txt`: "Test0 ori0 vertical, Y 3x5, k=Y height=3; horizontal rule width=3*5+2=17, transpose -> 17x3. Seems output given."
