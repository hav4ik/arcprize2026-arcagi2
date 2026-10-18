# E029 close reading: what V4.1-Flash-1017 does differently on the tool harness ($0, offline)

OLD = V4.1-Flash (E009 sub80s40), NEW = V4.1-Flash-1017 (E029 sub80s40). The harness, prompt (`lean2sub`), 80k budget, 40k soft-target notes and 60k "submit now" note are the same in both. This file covers reading only; the regex and statistics side is in `strategy_quant.md`.

## Summary
- **NEW does not think in a new way. It searches the same way, with fewer dead ends, and commits to code sooner.**
  - Over 30 paired traces, it tries about as many rules (7.7 against 8.4 per trace) and does about as much of its search in prose (16/30 traces drop rules in prose, against 20/30).
  - Its candidates reach code earlier:
    - first train-wide `solve()` at 28.7k against 33.7k;
    - in the SAVE pairs, 17.0k against 34.7k;
    - NEW never ran a candidate in 1/30 traces, OLD in 6/30.
  - It loops less: 13/30 traces have loops (about 4.0k each on average), against 24/30 (about 8.1k).
  - It has fewer very long turns. In the full runs, turns of 10k tokens or more hold 30% of output tokens, against 42%. Attempts with a turn of 15k or more fall from 35% to 20%.
- **Re-checking after a solution is not where NEW's tokens go.** From the first train pass to submit, NEW spends a mean of 1.4k tokens, of which about 1.0k is redundant checking (OLD: 4.0k). The remaining waste sits before the solution:
  - dead-end hypotheses argued in prose: about 7.3k per trace, in 22 of 30 traces;
  - reading printed grids by hand: about 5.2k, in 28 of 30;
  - long monologue turns: about 3.2k;
  - loops: about 3.0k;
  - simulating the rule in its head: about 2.2k.
- **Hard-pair gain.** NEW gets to a runnable candidate early enough that the 60k "submit now" note finds it with something to submit. OLD on the same pairs ran out of budget mid-monologue or submitted a hand guess. In the 8 GAIN pairs:
  - OLD: 4 of 8 ended with no answer, the test input, or a hand pick;
  - NEW: every trace submitted a code-produced answer. 6 of 8 had passed all train pairs; 2 submitted a near-fit rule under budget pressure.
  - The decisive steps were usually a perception reframing seen in a table that code printed (4 of 8), or a test-aware implementation (3 of 8).
- **NEW's regressions** (4 pairs):
  - two fast overfits: a rule that fits train but breaks on the test, with no test audit;
  - two runs that hit the 40-turn cap without an answer. NEW takes more, shorter turns.
- **Ranked proposals** (details in §4):
  1. Fix the turn cap: a turns-left note plus automatic `answer` capture.
  2. A one-line, test-only audit after train passes.
  3. A per-turn cap of 12–16k with a "test it in code now" continuation.
  4. A code-first nudge after grid prints, such as a compact-view helper.
- **Do not** gate submission on train-check, add a "stop re-verifying" line, or tighten the budget notes. Each would cut accuracy for at most about 1–2k tokens per attempt.

## Method and sample
- **Selection rule** (`reading/select_pairs.py` → `reading/selection.json`): one pair per task, and the same attempt index for OLD and NEW where possible (attempt 0 first).

  | Group | Pairs | Rule |
  |---|---|---|
  | SAVE | 12 | Every OLD and NEW attempt correct; ranked by mean OLD tokens minus mean NEW tokens |
  | GAIN | 8 | OLD 0/3, NEW solved; pairs where NEW went 2/2 first |
  | WASTE | 6 | Largest NEW correct attempts above 50k |
  | REGRESS | 4 | NEW 0/2, OLD solved; most OLD successes first |

- **Rendering:** `reading/render_pairs.py` (adapted from E026 `ds_render.py`) adds interpolated `[~tok N]` positions.
- **Readers:** one structured annotation per trace plus one COMPARE line per pair, following `reading/annotation_guide.md` (fields as in E026, plus post-pass, budget-reaction, loop and waste codes).
  - Annotations: `reading/{A,B,C,D,H,I,J}.jsonl` and `reading/K_*.jsonl`, 90 lines.
  - Aggregation: `reading/agg_annotations.py`.
  - Mechanical cross-checks: `reading/trace_mech.py` (60 traces) and `reading/turn_sizes.py` (all 835 attempts).
- **Caveats:**
  - **The sample is biased toward hard pairs.** NEW means 38.2k tokens per trace here, against 31.7k over the full run.
  - Token figures are reader estimates. Waste categories overlap: a 30k monologue can be both `prose_before_code` and `dead_end_prose`. Treat sums as upper bounds.
  - **No verbatim quotes of reasoning.** A safeguard stopped readers (both Opus and Sonnet) that copied reasoning passages, and several had to be re-run. Evidence below is therefore paraphrased with token positions. The first-batch files (A, B, C, D, H, I, J) still hold a few short verbatim excerpts.

| Group | Pair | Attempt (OLD/NEW) | Pair accuracy, OLD → NEW | OLD tokens (k) | NEW tokens (k) | First solve() (k), OLD / NEW | Train pass (k), OLD / NEW |
|---|---|---|---|---|---|---|---|
| SAVE | 13e47133_p1 | a0/a0 | 3/3 → 2/2 | 28.9 ✓ | 14.2 ✓ | 25.1 / 12.5 | 25.1 / 12.5 |
| SAVE | eee78d87_p0 | a0/a0 | 3/3 → 2/2 | 57.6 ✓ | 22.7 ✓ | 57.3 / 21.4 | 57.3 / 21.4 |
| SAVE | 6e4f6532_p1 | a0/a0 | 3/3 → 2/2 | 59.4 ✓ | 29.5 ✓ | 46.5 / 26.8 | 47.1 / 26.8 |
| SAVE | 332f06d7_p0 | a0/a0 | 3/3 → 2/2 | 35.1 ✓ | 11.7 ✓ | 34.4 / 10.4 | 34.4 / 11.1 |
| SAVE | 142ca369_p0 | a0/a0 | 3/3 → 2/2 | 70.6 ✓ | 48.6 ✓ | – / 40.3 | – / 40.3 |
| SAVE | 409aa875_p0 | a0/a0 | 3/3 → 2/2 | 63.6 ✓ | 18.2 ✓ | 59.3 / 12.8 | 62.8 / 16.2 |
| SAVE | db0c5428_p0 | a0/a0 | 3/3 → 2/2 | 50.1 ✓ | 18.4 ✓ | 6.4 / 14.9 | 49.9 / 17.8 |
| SAVE | 135a2760_p0 | a0/a0 | 3/3 → 2/2 | 33.4 ✓ | 16.3 ✓ | 21.6 / 15.6 | 32.5 / 15.6 |
| SAVE | 136b0064_p0 | a0/a0 | 3/3 → 2/2 | 40.6 ✓ | 20.6 ✓ | 39.5 / 20.4 | 39.5 / 20.4 |
| SAVE | e8686506_p0 | a0/a0 | 3/3 → 2/2 | 57.9 ✓ | 11.5 ✓ | 56.1 / 11.2 | – / 11.2 |
| SAVE | 8e5c0c38_p1 | a0/a0 | 3/3 → 2/2 | 10.8 ✓ | 7.6 ✓ | 10.1 / 6.7 | 10.1 / 6.7 |
| SAVE | d8e07eb2_p1 | a0/a0 | 3/3 → 2/2 | 30.4 ✓ | 11.9 ✓ | 25.9 / 10.6 | 29.7 / 11.6 |
| GAIN | 16b78196_p0 | a0/a0 | 0/3 → 2/2 | 67.6 ✗ | 47.9 ✓ | 44.3 / 47.6 | 47.7 / 47.6 |
| GAIN | 271d71e2_p0 | a0/a0 | 0/3 → 2/2 | 75.0 ✗ | 49.9 ✓ | – / 46.8 | – / 48.4 |
| GAIN | 3a25b0d8_p0 | a0/a0 | 0/3 → 2/2 | 43.2 ✗ | 26.8 ✓ | 25.9 / 17.6 | 25.9 / 20.6 |
| GAIN | 20a9e565_p1 | a1/a1 | 0/3 → 1/2 | 80.0 ✗ | 67.5 ✓ | – / – | – / – |
| GAIN | 21897d95_p0 | a0/a0 | 0/3 → 1/2 | 69.9 ✗ | 69.1 ✓ | 67.8 / 44.3 | – / – |
| GAIN | 446ef5d2_p0 | a1/a1 | 0/3 → 1/2 | 73.0 ✗ | 58.1 ✓ | – / 49.7 | – / 57.9 |
| GAIN | 7b5033c1_p0 | a0/a0 | 0/3 → 1/2 | 1.1 ✗ | 3.6 ✓ | 1.0 / 3.4 | 1.0 / 3.4 |
| GAIN | 800d221b_p0 | a1/a1 | 0/3 → 1/2 | 63.4 ✗ | 20.1 ✓ | 20.6 / 18.4 | – / 19.5 |
| WASTE | b6f77b65_p1 | a0/a0 | 0/3 → 1/2 | 62.7 ✗ | 75.7 ✓ | 31.1 / 27.5 | – / – |
| WASTE | 8b7bacbf_p1 | a0/a0 | 2/3 → 1/2 | 80.0 ✗ | 72.5 ✓ | 69.5 / 64.0 | – / 71.6 |
| WASTE | a6f40cea_p0 | a0/a0 | 2/3 → 1/2 | 80.0 ✗ | 69.0 ✓ | – / 65.5 | – / 68.6 |
| WASTE | 64efde09_p0 | a0/a0 | 1/3 → 2/2 | 37.6 ✓ | 68.2 ✓ | 35.1 / 66.2 | 35.1 / 66.2 |
| WASTE | 581f7754_p1 | a1/a1 | 3/3 → 2/2 | 42.0 ✓ | 64.3 ✓ | 13.4 / 17.7 | – / – |
| WASTE | a32d8b75_p1 | a0/a0 | 2/3 → 2/2 | 50.6 ✓ | 63.0 ✓ | 41.3 / 60.1 | 49.6 / 62.9 |
| REGRESS | 3dc255db_p0 | a0/a0 | 2/3 → 0/2 | 43.5 ✓ | 16.9 ✗ | – / 14.7 | – / 16.7 |
| REGRESS | 78332cb0_p0 | a0/a0 | 2/3 → 0/2 | 32.9 ✓ | 37.0 ✗ | 20.5 / 15.4 | 31.9 / 35.1 |
| REGRESS | 0934a4d8_p0 | a1/a1 | 1/3 → 0/2 | 65.3 ✓ | 40.3 ✗ | 6.2 / 6.6 | 65.2 / – |
| REGRESS | 195c6913_p1 | a2/a0 | 1/3 → 0/2 | 63.2 ✓ | 63.9 ✗ | 49.5 / 63.9 | 53.5 / – |

"First solve()" and "train pass" are reader-marked cumulative token positions; "–" means the trace never reached that point. In 6e4f6532 OLD, the "train pass" used per-figure hard-coded stamps.

## 1. What NEW does differently (30 OLD / 30 NEW traces)

| Behaviour | OLD | NEW |
|---|---|---|
| Correct in sample | 19/30 | 26/30 |
| Output tokens, mean | 52.3k | 38.2k |
| Tokens before the first tool call, mean (median) | 3.3k (0.2k) | 1.6k (0.1k) |
| …in SAVE pairs | 6.8k | 2.9k |
| First train-wide candidate `solve()`, mean | 33.7k | 28.7k |
| …in SAVE pairs | 34.7k | **17.0k** |
| Never ran a candidate on train | 6 | **1** |
| Final rule passed all train pairs | 18 | 24 |
| Core idea first appears, mean | 17.6k | 13.4k (SAVE: 16.6k → **7.0k**) |
| Core idea first appears, % of trace | 35% | 34% (unchanged: NEW is shorter *and* earlier) |
| Rules tried per trace | 8.4 | 7.7 (SAVE: 8.6 → 5.4) |
| Wrong rules dropped after a code test / in prose | 8 / 20 | **13** / 16 |
| Traces with loops (re-deriving, oscillating) | 24 (8.1k per trace) | **13 (4.0k)** |
| Hand-reading of printed grids | 9.2k | 6.6k |
| Perception mode (mixed / hand / code) | 25 / 4 / 1 | 27 / 2 / 1 |
| Tokens from first train pass to submit | 4.0k | **1.4k** |
| Redundant re-checks: count / tokens | 0.9 / 1.5k | 0.8 / 1.0k |
| Final answer is `solve(test_input)` after a full train pass | 16 | 21 |

**Full-run turn structure** (`reading/turn_sizes.py`, all 501 OLD and 334 NEW attempts):

| | OLD | NEW |
|---|---|---|
| Share of output tokens in turns ≥ 10k | 42% | **30%** |
| Share of output tokens in turns ≥ 20k | 18% | **9%** |
| Longest turn per attempt, median | 11.8k | 8.5k |
| Attempts with a turn ≥ 15k | 35% | **20%** |
| Mean turns | 12.8 | 13.6 |
| Accuracy when the longest turn is < 10k / 10–20k / ≥ 20k | 0.82 / 0.64 / 0.43 | 0.78 / 0.70 / 0.50 |
| Out of budget with no answer (`output_budget`) | 27 (5.4%) | **3 (0.9%)** |
| Turn cap reached (`max_turns`) | 5 (54–73k tokens) | 5 (**29–64k**, 4 with no answer) |

**Strategy differences, with evidence**
- **Perception: same tools, better readings.**
  - Both print grids early and then read them in prose. Neither writes object or diff tooling much beyond printing.
  - NEW's readings more often land on the right unit:
    - **800d221b:** rooms bounded by the frame colour at 3.7k, and only frame cells change at 4.9k. The rule was visible in a table by 12.2k. OLD treated the task as inpainting from seed points and tried about 13 global rules.
    - **409aa875:** connected components at 2.8k, against OLD's prose search for a lattice up to 16.2k.
    - **136b0064:** NEW split collinear same-colour runs correctly at once. OLD misread twice, discarded two correct sub-rules (7.5k and 24.2k) and re-found them (18.3k and 36.2k).
    - **135a2760:** NEW read the test layout correctly (four tall strips) at 3.3k. OLD misparsed it as 16 panels, about 4.5k wasted.
- **Hypotheses: fewer resurrections and abandonments.**
  - OLD found the right idea and then dropped it in at least 5 of the 30 pairs (from the COMPARE lines):
    - **332f06d7:** hand-simulated the hardest pair at 18.2k, made an error, and spent about 13k on wrong theories.
    - **409aa875:** found the idea at 38.9k, dropped it, and settled on it at 54.3k.
    - **20a9e565:** considered the right crop at about 49k and rejected it.
    - **8b7bacbf:** found the concept at 11.5k and again at 64.8k.
    - **136b0064:** see above.
  - NEW did this once (195c6913, where it dismissed the seed cells). NEW drops more wrong rules by running them (13 against 8 traces).
- **Debugging and code use.** NEW turns an idea into a general search or solver sooner. OLD more often hand-derives the test or hard-codes per-example parameters:
  - **332f06d7:** one BFS placement graph over all pairs at about 10k.
  - **6e4f6532:** NEW wrote a generic search solver (pass at 26.8k). OLD hand-built the test with off-by-one errors and then spent about 12k repairing it.
  - **16b78196:** an exhaustive exact-cover search with a unique solution.
  - **21897d95:** NEW started implementing at 39k instead of 62k, which left room for about 6 diff-driven fix rounds.
- **Verification.** NEW passes train in one shot more often and submits within a median of under 1k tokens. Post-pass work is mostly useful:
  - **142ca369:** a code audit of the test showed the literal train-passing simulator misfires on the densely packed test (about 8k).
  - **16b78196:** about 7k validating the search on train.

  OLD more often runs long post-pass doubt aimed at the wrong risk. In 3a25b0d8 it spent 17.3k on a multi-colour hole while missing holes its own printed output left empty.
- **Budget notes.**
  - Both ignore "Past the target" notes in nearly every trace that passes 40k. Of the 14 NEW traces over 40k, 12 ignored them; 2 (446ef5d2, 16b78196) kept working in short code turns that were productive.
  - The 60k "Budget nearly spent" note converts: 7 of those 14 NEW traces submitted within one or two turns of it, and all 7 were correct.
  - A note only arrives with tool output. 57% of NEW attempts that pass 40k cross it inside a turn of 5k or more, a mean of 7.9k past the mark before any note can reach them. Examples: 64efde09's 27k turn from 39.0k to 66.2k, and 581f7754's 26k turn.
- **Final answer.**
  - NEW: 21/30 submitted a fully train-passing `solve(test_input)`.
  - Of NEW's 26 correct traces, **6 were not a full train pass**:
    - 3 with a failing pair: b6f77b65 passed 2 of 5, 21897d95 3 of 4, 581f7754 had ex1 failing;
    - 64efde09: train passed only with hand-supplied palettes;
    - 20a9e565: a generator fitted to the test alone;
    - 142ca369: a straight-ray special case of the simulator.

## 2. Where NEW traces still waste tokens
Reader-tagged categories, mean tokens per NEW trace over all 30, and the number of traces with the pattern (≥ about 1k). Overlapping, so upper bounds.

| Pattern | Tokens per trace | Traces | Typical form (paraphrased, positions in that trace) |
|---|---|---|---|
| Dead-end hypotheses in prose | 7.3k (20.7k in WASTE) | 22 | Searches for a formula that does not exist, where a ten-line enumeration would decide (see examples below) |
| Hand-reading printed grids | 5.2k | 28 | Re-reads coordinate lists or rows the tool just printed. 20a9e565: about 20k reading the test arches (41–64k). 21897d95: about 18k on block layouts |
| Monologue turns (long no-tool turns, incl. before the first call) | 3.2k | 9 | 142ca369: one 33.5k turn (5.5–39.1k). 136b0064: 19.6k before the first call. 16b78196: a 16.9k turn. 581f7754: 26k and 9k turns. eee78d87: about 11k before the first call |
| Loops (re-deriving or oscillating) | 3.0k (9.7k in WASTE) | 9 | b6f77b65: about 18k re-deriving per-piece shifts (28.6–72.7k). 195c6913: about 20k on skeleton variants. 0934a4d8: about 10k. 64efde09: about 7k flip-flopping between bar faces |
| Simulating the rule in its head | 2.2k | 15 | Traces reflections, rays or block paths by hand before coding. b6f77b65: about 12–15k (12.7–27.3k). 142ca369: about 9k |
| Code-bug churn | 1.1k | 7 | 21897d95: about 14k of segmentation fixes (39–67k), which bought the solve |
| Hand check of a test output the code made | 0.5k | 9 | 1–2k each (409aa875, 6e4f6532, b6f77b65) |
| Post-pass re-verification and alternative-rule doubt | 0.3–1.0k | 4 | Small; often useful (142ca369, 16b78196) |
| Restating the rule | 0.3k | 5 | Small |

Dead-end prose examples:
- **64efde09:** about 29k on offset formulas and palette sort keys. It never noticed that the dots set the offsets; OLD did, and finished at 37.6k.
- **a6f40cea:** about 38k of prose about one anomalous object. A transform search at 54k settled it.
- **a32d8b75:** about 16k comparing masks by eye (13.5–29.5k), then about 14k on placement theories. A single brute-force call settled it at 36.5k; OLD brute-forced at 32k.

The answer to the question "is NEW's waste post-solution re-verification?" is no. Within the reading, the recurring sinks are:
- **Prose hypothesis testing instead of a code probe.** Dead-end prose, loops and in-head simulation together come to about 12k per trace, and 30–45k in WASTE traces.
- **Re-reading printed output by hand:** about 5k.
- **Monologue turns that the budget notes cannot reach.**

In SAVE pairs, readers put NEW's waste at about 3.5–11k in most traces, mostly hand-reading (3.4k mean). The exceptions are monologues of 19–34k in 136b0064 and 142ca369; 5 of 12 SAVE traces have a monologue turn.

## 3. What explains the hard-pair capability gain
From the 8 GAIN pairs, plus the 3 WASTE pairs where OLD failed:
1. **Earlier commitment to runnable code leaves budget to finish.**
   - In GAIN, NEW's longest turn averages 10.6k against OLD's 20.3k. NEW's idea comes at 13.4k against 20.8k, and its first solver at 32.5k with 7 of 8 traces running one (OLD: 5 of 8).
   - OLD's failures on the same pairs, from the annotations:
     - 4 of 8 ran out of budget or turns: 20a9e565, 800d221b, 446ef5d2 and 21897d95. 21897d95 had the right rule in prose but coded it only at 62k+.
     - 271d71e2 had the decisive count table at 25.7k but never coded a solver.
   - NEW reaches the 60k "submit now" note holding a candidate. 21897d95, 20a9e565, 8b7bacbf, a6f40cea and b6f77b65 were all submitted right after that note and all correct.

   This matches the README: the gain needs the 80k budget and disappears at 60k.
2. **Data-driven reframing from code-printed tables (4/8):**
   - 271d71e2: printed change sets per box showed that the number of changed cells equals the shift distance (12.6k);
   - 800d221b: rooms;
   - b6f77b65: a colour-count script showed at 8.1k that the marker colour is deleted;
   - 446ef5d2: the 3-cell patch is the test marker, so the area is 220 = 10×22.

   OLD on these pairs framed the problem wrongly and defended the framing in prose.
3. **Test-aware, structurally exact implementations (3/8):**
   - 7b5033c1: NEW looked at the test first and saw the chain bend back, so path order differs from reading order. OLD never looked at the test.
   - 16b78196: block plus pieces must form a solid rectangle, an exact cover with a unique solution. OLD's max-overlap rule left three ambiguous assemblies.
   - 3a25b0d8: NEW used nearest-cell region correspondence; OLD's overlay left holes empty on a test figure of a different size.
4. **Pragmatic near-fit submission** (21897d95 3/4, b6f77b65 2/5, 20a9e565 test-only). Under the 60k note NEW submits its best partial rule instead of a hand guess, and the test needs only the cases that rule handles. This is partly luck, but it is a behaviour, and a train-check gate would destroy it.

What does **not** explain the gain:
- the number of hypotheses (6.4 against 8.2 in GAIN, comparable at that difficulty);
- hand perception volume (NEW still hand-reads 8.6k per GAIN trace);
- verification.

**The cost of the same efficiency shows in the regressions:**
- **3dc255db:** NEW used 16.9k. It accepted a proxy rule that fits train (the largest centroid-to-bbox distance) without asking whether it describes the shapes, and missed the one test object where the proxy disagrees.
- **78332cb0:** NEW fitted a spurious sort key and assumed the orientation without testing it. OLD found the edge-matching chain.
- **0934a4d8:** NEW averaged about 1.0k tokens per turn and hit the 40-turn cap at 40.3k with no answer set. It also used only whole-grid pass/fail tests, where OLD's per-cell mismatch map exposed the band-local transpose.
- **195c6913:** NEW dismissed the seed cells, then looped on skeleton variants and hit the turn cap at 63.9k.

## 4. Proposed harness and prompt changes, ranked
Each proposal is testable on the 42 hard pairs from the README (old correct runs needed more than 40k) or on the full 167 pairs. Use paired comparison against this run (`evals/paired_vs_baseline.py`).

1. **Turn-cap safety net**
   - **Change:**
     - When 8 turns remain, append to the tool output: "N turns left: set `answer` to your best `solve(test_input)` now."
     - Have the harness auto-capture the last successful `solve(test_input)` (or any grid-valued `answer` candidate) as the fallback.
     - Optionally raise `max_turns` from 40 to 60, since output tokens remain the binding budget.
   - **Evidence:**
     - NEW takes more, shorter turns (13.6 against 12.8).
     - 5/334 NEW attempts stopped at the turn cap at 29–64k tokens, 4 of them with no answer.
     - The 23 attempts with ≥ 30 turns score 0.26.
     - 0934a4d8 hit the cap at 40.3k and never saw the 60k note, the one note that reliably converts.
   - **Gain:** about 1–2 correct attempts per 334 (+0.3–0.6 pp), at no token cost.
   - **Risk:** very low. Bounded by the 80k cap.
2. **One-line test audit after train passes**
   - **Change:** a prompt line such as: "When train passes, run one code check that each test object or region satisfies the assumptions your rule relies on (shape, orientation, count, colours not seen in train); fix the rule if one does not."
   - **Evidence:**
     - 2 of 4 regressions (3dc255db, 78332cb0) are train-fitting proxies that a test-object check would flag.
     - The gains 7b5033c1 and 3a25b0d8 come from exactly this kind of look.
     - NEW's post-pass phase is only 1.4k, so there is room.
   - **Risk:**
     - +1–2k tokens per attempt.
     - E017's `lean2roles` (which included a test audit) did not raise accuracy on OLD (40% → 47% audit rate, no accuracy gain), so ask for one code check, not a prose essay.
     - Test it alone.
3. **Per-turn output cap of 12–16k with a "test now" continuation**
   - **Change:** `max_tokens` per turn of 12–16k. On truncation, the next user/tool message says: "Your turn was cut at N tokens without running code. Write the smallest code experiment that tests your current hypothesis." Check first that the DeepSeek API returns truncated `reasoning_content` and accepts it in the next request.
   - **Evidence:**
     - Accuracy falls with the longest turn (0.78 → 0.70 → 0.50).
     - Budget notes cannot reach a monologue (57% of NEW attempts over 40k cross it inside a ≥ 5k turn).
     - The WASTE traces' largest sinks are prose stretches of 15–33k (64efde09, 581f7754, 142ca369, 16b78196).
     - For Qwen, `--turn-think-cap 8000` gave +7.2 pp (E020).
   - **Gain and risk:**
     - Direct saving is small: only 1.4k (16k cap) to 2.5k (12k cap) tokens per attempt lie above the cap.
     - The value is in redirecting search, and the risk is real: 142ca369's 33.5k turn and 16b78196's 16.9k turn produced the decisive rule.
     - Start at 16k (17% of attempts affected, which score 0.63).
4. **Code-first probing nudge**
   - **Change:** a compact grid viewer in the session (for example `--helpers`' `show`), or a prompt line: "When two readings of an example conflict, enumerate or score the candidates in code (match counts, per-cell mismatch maps) rather than comparing by eye."
   - **Evidence:**
     - Hand-reading is in 28/30 traces at about 5k, and dead-end prose in 22/30 at about 7k.
     - a32d8b75, a6f40cea and 64efde09 each lost 15–30k before a single brute-force call settled the question.
     - 0934a4d8 failed for lack of a partial-agreement measure.
   - **Risk:** brevity prompts hurt OLD (E009 `lean2brief`: −8 to −14 pp at 96k and 64k). Word it as "use code to decide", not "be brief", and test it on its own.
5. **Keep the budget notes as they are.**
   - "Past the target" is ignored, but the 60k note converts, and the hard-pair gain needs 60–80k.
   - Tightening `--budget-warn` or the soft target trades the GAIN pairs for tokens. That is the s20x60 result: +1.2 pp pass@2, within noise.
   - Removing the soft-target text is harmless but saves nothing.

**Not recommended:**
- **A train-check-gated submit.** It would have rejected 6 of NEW's 26 correct answers in this sample (3 near-fit, 3 test-specific). Use `train_check` for ranking only (E017 `ranked_answers`).
- **A stop-on-verified or "do not re-verify" rule.** Mean post-pass work is 1.4k and is often useful. This is also part of what `lean2brief` did.

## Files
- **Scripts** (in `reading/`; run with `.venv/bin/python -I <script>`):

  | Script | What it does |
  |---|---|
  | `select_pairs.py` | Selects the 30 pairs → `selection.json` |
  | `render_pairs.py OUTDIR` | Renders the traces for reading. Renders were in `/tmp/e029_render`, not kept |
  | `agg_annotations.py` | All annotation aggregates in §1–3 |
  | `trace_mech.py` | Mechanical metrics for the 60 traces → `trace_mech.json` |
  | `turn_sizes.py` | Full-run turn and stop statistics |

- **Annotations:** `reading/*.jsonl` (30 pairs × OLD/NEW/COMPARE) and `reading/annotation_guide.md`.
