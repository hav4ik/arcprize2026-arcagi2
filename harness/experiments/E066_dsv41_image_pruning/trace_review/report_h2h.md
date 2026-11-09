# Head to head: E72-152 vs candidate 1 (c1x), image runs

Traces: h2h/e72 (23 attempts), h2h/c1 (12 attempts), h2h/ref (full model, 4 tasks). Dumped 2026-10-29, evals still running.
SMALL SAMPLE: 8 tasks finished by both (14 c1 / 16 e72 attempts, only easy early finishers). Nothing below is statistically meaningful.

## Common tasks (1818057f 2ba387bc 38007db0 45a5af55 58490d8a 7b5033c1 8f215267 bf45cf4b)
| | c1 | e72 |
|---|---|---|
| attempts finished (common) | 12 | 16 |
| attempts correct | 11/12 | 15/16 |
| out tokens, mean per attempt | ~3.6k | ~4.9k (higher on 38007db0 / 45a5af55 / 8f215267, same pairs) |
| tool errors (total, common tasks) | 7 | 9 |
Note c1 has only one attempt on 6 of the 8 tasks (the other is unfinished = likely harder/longer, so c1's finished set is biased toward quick ones).

## Observations
- Both: terse reasoning, run code on the train pairs before submit_answer in essentially every trace; both submit via solve(test_input)/answer; no rescues, no loops, all stop=submitted, no budget/turn-cap endings.
- Tool errors in both are of the same sort: shape/inhomogeneous-array bugs and typos. Typos seen: c1 `c.mmost_common` (7b5033c1_a0), e72 `np.conconcatenate` (981571dc_a1, non-common task). Both self-repaired in the next turn. Doubled-token typos occur in both; no sign of garbled output.
- Image use: neither trace refers to the image in a way that carries the solution; reasoning is driven by JSON + code (c1 and e72 mention "image" in the prompt text only). No evidence either exploits the picture.
- e72 is somewhat more verbose and more exploratory (e.g. 38007db0: 13 turns/6.5k tok vs c1 10 turns/4.8k) but the reference also does long analysis there. No evidence it is worse mechanically.

## Failure table (all wrong attempts in either candidate, both attempts counted)
| task | att | cand | cause | pruning-related? | full model solved? |
|---|---|---|---|---|---|
| 7b5033c1 | 0 | c1 | wrong hypothesis: order colours by first (top) row, repeat by count (output 8,5,6,6). Fits both train pairs, wrong on test (test has two chains) | no (hard/ambiguous task) | no (ref a0 and a1 both wrong; ref a0 gave the same answer) |
| 7b5033c1 | 0 | e72 | wrong hypothesis: row-major flatten of non-bg cells. Fits train, wrong on test; 764 tokens, 2 turns, no test-assumption check | unclear (shortcut, but ref also fails the task) | no |
| 31f7f899 | 1 | e72 | wrong hypothesis on test: sorts per-column heights; test has 3-wide bars and a bar at the border. Model noticed "wide bars" and then argued the answer coincides, submitted without checking | unclear, mild yes (weak verification of an out-of-range test assumption) | yes (ref a0 and a1 both correct) |
c1 has no attempt on 31f7f899 yet. Both 7b5033c1 a1 attempts (c1, e72) were correct (c1 with 3 tool errors, e72 with 0).

## Quotes
- c1/7b5033c1_a0.txt: `bg=c.mmost_common(1)[0][0]` -> `AttributeError: 'Counter' object has no attribute 'mmost_common'`, fixed next turn ("Typo most_common").
- e72/981571dc_a1.txt: `res.append(np.conconcatenate([[sep], tiles[idx][rr], [sep]]))` -> `module 'numpy' has no attribute 'conconcatenate'`.
- e72/31f7f899_a1.txt: "Wait test bars: gray 5 occupies columns 2,3,4 each height7 — that's a wide bar ... Coincidence but output matches likely."

## Inference
- Candidate health: both look mechanically healthy (no loops, garbling, caps). e72 is at least not worse: 15/16 vs 11/12 on the common set, one extra failure only on a task c1 hasn't finished. Slightly longer traces. Typo rate per attempt looks similar. Slight lean to e72 on the held-out loss; the trace data cannot separate them.
- Systematic difference: e72 uses more tokens/turns on the same pairs; no accuracy gap visible. A fair comparison needs the full 120 tasks.
- The one failure with a full-model success (31f7f899 a1 e72) is a judgment lapse on out-of-train-range test input, not a mechanics failure; one sample.
