# E003 Part 2: Failure analysis of DeepSeek traces (2b83f449, 5545f144)

Scope: 8 traces (Flash = V4.1-Flash, Pro = V4-Pro; thinking, effort max; ARC Prize plain prompt). Each was read in full. Offsets are approximate char positions.

## The two puzzles

**2b83f449** (Flash 1/2, Pro 0/2).
- **Part A:** each `777` in a 0-row becomes `868`, and the 8-rows above and below get a 6 at its centre. All 4 attempts got this right within the first 3–5% of the trace.
- **Part B:** after the bars are drawn, the 8/3 cells form a maze (0s and 6s are walls; the `868` flanks are passages). Each input 3 is a ball, and the number of 3s in each connected region is conserved (I checked every pair).
- The balls settle in the region's bottom segment, at the end opposite the passage they entered through. So `33` means 2 balls; segment length is irrelevant.
- Flash a1 found exactly this. Every error in the other attempts is a 3-placement error.

**5545f144** (0/4). Train panels are time-ordered frames.
- An oriented shape walks a trail of dots: straight to the next dot, turn, eating each one. It points in its direction of travel with its head on the dot.
- The output is the final frame.
- The test is a single 15×15 frame with no separators. The 12-dot walk ends pointing left at (6,4), so the shape appears rotated 90° counter-clockwise. A straight-first simulation reproduces the ground truth exactly.

## Per-trace summary

| Trace | Wrong cells | Chars | Commits at | Failure class |
|---|---|---|---|---|
| Flash 2b83 a1 | 0 ✔ | 179k | 104k / 135k (75%) | (principled success) |
| Flash 2b83 a0 | 2 (row 18) | 218k | 91k; last row guessed at 213k (98%) | wrong rule (global count formula) + guess on one row |
| Pro 2b83 a0 | 9 (rows 2/8/14/18) | 193k | 123k (64%) | wrong rule (overfit proxy: "edge fill if gap ≤3") |
| Pro 2b83 a1 | 6 (rows 2/8/14/18) | 282k | 178k (63%) | right side-rule, wrong doubling feature ("distance ≤4"); conservation violation ignored |
| Flash 5545 a0 | 12 | 210k | 189k (90%) | wrong rule + guess ("go to topmost isolated pixel"); uncaught misread |
| Flash 5545 a1 | 9 | 232k | 220k (95%) | wrong rule; guessed by copying Ex0's output position |
| Pro 5545 a0 | 10 | 223k | 187k (84%) | gave up: "keep shape, delete dots", which it knew fails 3/3 train pairs |
| Pro 5545 a1 | 12 | 223k | 177k (80%) | wrong rule with free parameters (rotation and anchor fitted per pair) |

Diffs against the truth:
- **Flash 2b83 a0:** row 18 predicted `…880…833`, truth `…830…883`. It missed the 3 beside the hole and doubled the edge 3.
- **Pro 2b83 a0/a1:** both wrote `88`/`38` instead of `33` on the long segments in rows 2, 8 and 14, plus the same row-18 errors.
- **3-counts:** the truth has 18. Pro a0 output 13 and Pro a1 output 16, both breaking conservation; both Flash attempts output 18.
- **Flash 5545 a0 and Pro 5545 a1:** both predicted the shape unrotated at rows 1–3. That is step 1 of the 12-step walk, reached by coincidence.

## Thinking-process findings

**1. Perception is not the bottleneck.**
- Grid lines are only 1–3% of the text; the traces are prose with `c0..cN`-indexed rows.
- Slips were rare and mostly caught: Flash 2b83 a0 dropped a test row (caught at 99k).
- Two Flash 5545 misreads were never caught:
  - a0 decided Ex0 row 7 is a "typo" (2.9k) and lost the frame-3 V.
  - a1 split the test's 6-cell shape into "two V shapes" (84k). That triggered its copy of Ex0's answer.

**2. Search is broad, shallow and has no memory.**
- 45–65% of each trace is one-row correlation tests (gap lengths, parity, lasers, Game of Life, "search memory"), each dropped after one counterexample.
- Ideas are re-derived 3–5 times. Hedging is heavy: "maybe" appears 244–859 times per trace.

**3. The correct idea appears, then gets dropped.** All four 5545 traces raised "panels are animation frames": Flash a1 at 25k, Pro a1 at 37k, Flash a0 at 59k ("Unlikely."), Pro a0 at 70k.
- **Flash a1** got within one step at 94k: "moves in the direction its stem points… A→B up… D→output down. Yes!" It then tested Ex2 with panel-level displacement instead of dot-to-dot moves, and dropped the idea.
- **Pro a0** separately logged falling dot counts (8→7→6→5, 160k) and an "anchor dot common to all panels" (181k). It never combined them.

**4. Contradictions are noticed, then rationalised.** Every wrong attempt saw its rule fail:
- Flash 2b83 a0 explained Ex0's misfits with an invented "global deficit" (180k).
- Pro 2b83 a0: "matches all training except bottom default extras… There is no reason to only produce singles" (168–169k). This locks in its row-18 error.
- Pro 2b83 a1 computed that `33` on rows 2, 8 and 14 "exactly adds 3", which is the correct fix (247k). Then: "Let's not over-stress about count" (260k).
- Pro 5545 a0: "the rule is not just noise removal… Oops." (214k). It submitted noise removal anyway.

**5. Pseudo-verification.**
- Pro 5545 a1's "This is consistent!" (177k) picked a different component, angle and endpoint for each pair.
- Pro 2b83's thresholds (≤3, ≤4) fit every train row but are causally wrong.
- Verification is ≤10% of wrong traces and 22% of the correct one.

**6. Self-imposed stopping.**
- No trace hit a limit: all stopped normally at 61–88k tokens.
- Yet "Given the time" (5× in Flash 5545 a0) always comes right before a guess.
- Commitment comes late (63–98%), and no test output was checked against the train invariants.

**7. Format shift (5545).** About 20% of each trace hunted for hidden panels in the test. Pro a0 rejected the frame idea as "too complex for test without panels" (182k).

## 2b83f449: what Flash a1 did differently

Three attempts found 3-count conservation early (Flash a0 39k, Pro a1 36k, Flash a1 64k); Pro a0 never did.
- **Flash a1** turned it into a mechanism: "3s 'fall' through the maze under gravity" (80k). That gave conservation per component ("Perfect!", 104k).
- It then checked all 17 train components and let one failure refine the fill side (135k). Its row 18 follows mechanically: 4 components with 1 ball each.
- **Flash a0** kept a global count formula plus fudges.
- **Pro a1** found the correct side rule (bottom tip of the cross from above, 145k), but let a spurious distance feature override conservation.

## Flash vs Pro

- **Speed and accuracy:** Pro is 2–3× slower (639–889 s vs 271–337 s) at similar token counts, and was not better here (0/4 vs 1/4).
- **Style:** Pro hedges more and verifies more systematically, but what it verifies are overfit proxies. Flash guesses sooner. Flash's one success came from a causal model, not from more checking.
- **Consistency:** both Pro 2b83 attempts made the same doubling error, a model-level bias rather than sampling noise.

## Implications

**(a) Harness**
- **Sandbox plus a train-pair verifier requiring a single executable `transform()`.** It would catch 3 of the 4 wrong 5545 rules: Pro a0 fails 3/3, Flash a0 fails Ex2, and Pro a1's rule cannot be written as one function. It would not catch Pro 2b83's proxies, which pass on train.
- **Train invariants checked on the test prediction** (e.g. per-colour and per-component counts). This flags Pro 2b83 a0 (13 vs 18) and a1 (16 vs 18), and the per-component check also flags Flash 2b83 a0: all 3 wrong 2b83 attempts. Cheap and high-value.
- **Diff feedback on train pairs** would shorten the one-row-at-a-time hypothesis loop.
- **Object and panel extraction tools** would have prevented both uncaught Flash misreads.
- **Second opinion over the trace's own observations** (frames, stem direction, falling dot count, anchor dot), or a ledger of abandoned hypotheses, is the best bet for 5545.
- **Sampling:** Flash got 1/2 on 2b83, so diverse samples plus verifiers beat longer single traces.

**(b) Distillation**
- Use Flash 2b83 a1, but compress its 57k-char opening spiral to: colour-count invariants → causal/object model → per-component verification → construction.
- Exclude near-miss wrong traces. Their key moves are the rationalisations to avoid.
- Since perception is adequate, target the search policy:
  - Revise the rule when it fails instead of patching it.
  - Prefer mechanisms over thresholds.
  - Re-test ideas that were dropped untested.
  - Never let a different test format veto a rule that fits all train pairs.
