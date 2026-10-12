# E003 Part 1: DeepSeek reasoning traces on solved tasks (db695cfb, 8b9c3697)

Scope: 8 traces (Flash/Pro × a0/a1 × 2 tasks), all correct, all read in full (grid transcriptions skimmed). Offsets are character positions in the decompressed `.reasoning.txt`.

## Puzzle rules (ground truth)

- **db695cfb**: Two 1s on the same diagonal are joined by a 1-segment. A 6 on that segment stays 6 and fires an edge-to-edge 6 diagonal perpendicular to it. Other 1s and 6s are unchanged distractors. Test: 4 aligned 1-pairs, 4 trigger 6s, 6 distractors; the 6-lines never cross a 1-cell.
- **8b9c3697**: Each "comb" object has one short protrusion (central tooth or tab). A 2-block aligned with it, with the same cross-section width, slides in until it touches; swept cells become 0. Every other 2 is erased. Test: 3 objects, 3 movers, 8 distractors.

## Per-trace summary

| Trace | chars / out-tok / s | Rule first stated (offset, % of trace) | Phase split (hyp / test-analysis / build / post-check) | Digit-dense lines | "wait" | "maybe" |
|---|---|---|---|---|---|---|
| Flash db a0 | 60k / 31.9k / 107 | 4.5k (8%) | 27 / 9 / 40 / 24 % | 37% | 28 | 63 |
| Flash db a1 | 64k / 31.3k / 114 | ~8.5k (13%) | 25 / 25 / 31 / 19 | 28% | 27 | 67 |
| Pro db a0 | 147k / 60.6k / 516 | 10.9k (7%), confirmed 24k | 16 / 14 / 10 / **60** | 16% | 62 | 400 |
| Pro db a1 | 120k / 51.2k / 443 | 12.4k (10%) | 15 / 20 / 19 / **46** | 14% | 66 | 334 |
| Flash 8b a0 | 120k / 47.6k / 194 | 54.7k (46%), a proxy rule | 45 / 20 / 27 / 8 | 14% | 81 | 106 |
| Flash 8b a1 | 131k / 49.0k / 212 | 54.4k proxy; decided by analogy ~65k; "concavity" 124.8k (95%) | 49 / 20 / 19 / 12 | 13% | 88 | 138 |
| Pro 8b a0 | 197k / 68.6k / 660 | 60.1k (30%) "shortest protrusion" | 43 / 19 / 19 / 19 | 10% | 93 | 438 |
| Pro 8b a1 | 186k / 67.9k / 626 | 56.6k (30%) "protrusion, same cross-section width" | 30 / 24 / 13 / 33 | 11% | 103 | 391 |

"Digit-dense" = lines over 30% digits (grid rows, coordinate lists, per-cell `c7 6` lines). Every trace wrote the test rows 2–6 times (48–140 row lists for a 23–25-row grid).

### Hypothesis lifecycles

- **Flash db a0**: Fill between 1s, keeping 6s → "6s on 1-segment get extended along perpendicular diagonal" (4.5k) → checked on Ex0–3 → the isolated 1s in Ex3 are accepted as distractors → color precedence is considered (17.4k) and found irrelevant on the test. **It never looked at train pair 4** (counts: Ex0×17, Ex1×4, Ex2×3, Ex3×24, Ex4×0).
- **Flash db a1**: First guess "1s are propagated along diagonal away from nearest 6?" (1.5k) is refuted by Ex2 → pair-fill → perpendicular firing. Later it rejects joining same-row/column 1-pairs (44–47k).
- **Pro db a0**: First guess "each color draws its own diagonal direction" is refuted by Ex0 → pair-fill → blocker fires a perpendicular line (10.9k). After the grid was done (59k) it spent ~88k chars on unsupported alternatives (6-6 pairs, 1–6 shared diagonals, parity (128.1k: *"In all training examples, the only 6s whose orthogonal diagonal intersects the 1 line are exactly those on the 1 line"*), rejecting all, then plausibility counts (139.9k: *"35 ones + 60 sixes = 95 cells… plausible for ARC?"*).
- **Pro db a1**: Same path, starting from "extend every non-bg cell to the border".
- **Flash 8b a0**: "2s become 0 / extend" → ray-to-nearest-object (13.8k), refuted by many clear rays that were erased → hole-shape matching (22.7k) → destination inside the object bbox → *"contact cell(s)… must lie strictly inside the bounding box… (not on any boundary)"* (54.7k). A geometric proxy that happens to agree on this test; checked on ~7 of 18 train markers.
- **Flash 8b a1**: hole matching → alignment → perpendicular run-length match (54.4k). It then found a counterexample it could not resolve (55.5k: *"Marker height1 matches! But output removes marker r2 c1."*). It switched to **template analogy** (65.2k: *"Top-right object A = ex1 A (same shape)"*) to settle selection; "concavity/notch" (124.8k) was a post-hoc justification.
- **Pro 8b a0**: slide-until-contact (13.5k) → protrusion with matching cross-section (32.3k). Counterexample: ex1 r16 c4 sits under a long leg. **It re-transcribed ex1 to rule out a misread** (55.7k: *"Maybe I misread row indices"*) → *"shortest protrusion"* (60.1k) → checked against all 5 movers and listed the unused markers (64.6k).
- **Pro 8b a1**: Nearly the same path: "single protrusion with same cross-section width" (56.6k) → same r16 c4 counterexample (57.9k) → distance? → "short central protrusion" (~68k).

### Perception

All traces use 0-indexed (r,c). db traces convert to invariants s=r+c, d=r−c and solve intersections algebraically (r=(s+d)/2); 8b traces build component inventories (bbox, size). Errors were few and all self-caught within ~1k chars: Flash db a1 miscounted Ex3 ("3 1s and 5 6s"); Pro db a0 read the Ex3 1-line as stopping at (8,8); Pro db a1 called E(5,2) isolated, then found its partner 400 chars later; Flash 8b a1 had an off-by-one output row; Flash 8b a0 twice named the wrong nearest cell. None reached a final answer.

### Verification

No trace re-rendered a full training output cell by cell. db checks were object-level (which 6s fire, endpoints); 8b checks were per-marker (right granularity, uneven coverage; Flash 8b a1 proceeded with a known counterexample). The test output was always re-checked: a per-row "set union" table (Flash db a1), cell counts (Pro db a0), and repeated row-length counts everywhere.

### Output construction

Always: list non-background cells per row, then emit the row. Flash db a0 spelled out every cell (`c0 3 … c20 3`, 608 one-cell lines). No construction error survived; only mislabelled or double-counted rows.

## Cross-trace patterns

1. **Easy rule, expensive execution.** db695cfb: rule within the first 7–13% in every trace; the rest is applying it to a 25×21 grid and doubting it.
2. **Hard rule = counterexample-driven search.** 8b9c3697: the right intuition ("markers fill a hole/slot") appears early (4.8k–22.7k) but takes 55–60k chars to make exact, always via a per-marker moved/erased table.
3. **Priors invoked:** colors as roles (1 line / 6 blocker; 2 token / 0 trail); "ARC includes distractors"; task-designer intent; diagonal invariants; shape templates across pairs; "shortest/central protrusion is the socket".
4. **Correct answers from incomplete rules.** Both Flash 8b traces got the right output with a wrong or under-verified rule (bbox-interior proxy; analogy despite a counterexample).

## Flash vs Pro

- **Efficiency:** Flash uses 0.5–0.7× Pro's tokens and is 3–5× faster; telegraphic style ("Need…", "Good."). Pro hedges heavily (334–438 "maybe" per trace; "maybe maybe" up to 28×).
- **Pro's extra tokens** go mostly to post-construction reconsideration (46–60% of db traces), not to finding the rule, which Pro reached no sooner than Flash in absolute chars.
- **Rule quality:** on the hard task Pro has the true rule ("shortest protrusion, matching width"); Flash has proxies. Pro handles counterexamples better: it re-checks perception before dropping a hypothesis.
- **Rigor:** Pro is more thorough; every trace except Flash db a0 checked all 5 db train pairs.

## Attempt-to-attempt variation

- **db695cfb:** near-identical across attempts for both models; only the first wrong guess and the late alternatives differ.
- **8b9c3697 Flash:** different routes (bbox-interior vs run-length → analogy → concavity), same output.
- **8b9c3697 Pro:** both attempts converge on the protrusion rule via the same counterexample (r16 c4).

## Implications

**(a) Prompting / harness**
- **Python sandbox:** transcription, grid construction and row counting become tool calls (~30–45% of Flash traces, ~25–30% of Pro).
- **`verify(rule_fn)` on every train pair:** would end post-construction second-guessing (46–60% of Pro db traces) and catch the skipped Ex4 and Flash 8b's proxy/analogy rules.
- **Combined:** ~50–70% fewer tokens on these tasks, most for Pro.
- **Prompt:** require "state the rule as an algorithm, check all N train pairs" before building output; add a component/ray-hit helper for object tasks.

**(b) Distillation into a small model**
- **Imitate:** coordinate/invariant notation; component inventories; hypothesis → counterexample → refine loops on per-object/per-marker tables; "explain every distractor" checks; re-reading the input when a counterexample appears (Pro 8b a0); set-union row construction; Flash's terse register.
- **Prune:** per-cell enumeration; repeated full-row re-listings and length counts; "maybe maybe" hedging; post-hoc alternatives with no training support (6-6 pairing, parity).
- **Label by rule validity, not just correctness:** both Flash 8b traces would teach "answer by analogy despite an unexplained counterexample".
- **Best target:** Pro's hypothesis segments with Flash's execution style, trimmed to rule discovery plus one verification pass (~25–40k chars instead of 60–197k).
