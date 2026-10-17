# E026: how DeepSeek-Flash thinks on ARC-AGI-2, and what the task summary changes ($0, offline)

## Summary

**Data.** All runs are DeepSeek V4.1-Flash on the 120-task / 167-pair public eval set.

| Condition | Source | Attempts | Accuracy | Output tokens per attempt |
|---|---|---|---|---|
| NT: no tools, max effort | E001d attempt 2 | 167 | 51.5% | 56.4k |
| S80: tools, no summary | sub80s40 | 501 | 66.5% | 40.7k |
| S96: tools, no summary | sub96n | 334 | 67.7% | 45.7k |
| V1: S80 + `--task-summary` | E017 | 167 | 67.1% | 37.1k |
| V3: lean2roles + summary + helpers | E017 | 358 | 65.9% | 41.4k |

Method:
- A regex pass covered all 1,527 attempts.
- Readers annotated 31 traces in a structured way (`ds_annotations/`). 10 tasks were annotated in all three conditions (NT, S80 attempt 0, V1), plus one extra NT trace.
- **Sample bias:** the 10 tasks were chosen on purpose to mix S80/V1 outcomes, so their per-condition accuracies mean nothing.
- **Missing reasoning:** `arc2_deepseek-flash_{high,max}.jsonl` save only `reasoning_chars`, not the reasoning text. `arcprize_*_{low,high}_a1` hold just 7 and 3 traces.

**Does DeepSeek compute the summary's statistics itself when it has no summary? Yes, always, but mostly as cell-coordinate reasoning, not as tables.**
- **Without tools:** in 10/10 annotated NT traces it worked out by hand:
  - grid sizes, background, object lists with bboxes and sizes, and in→out object correspondences: 10/10 each;
  - cell diffs: 8/10;
  - colour counts and the test-vs-train comparison: 6/10 each.
  - Counting, dimension and object words appear in 98–100% of all 167 NT traces.
  - Hand perception takes about **33% of NT tokens** (annotated, token-weighted), about 25k tokens per trace.
  - It is not an up-front phase: lines that transcribe cells are spread evenly over the trace (9.6–10.3% in every decile). The model re-reads cells for each new hypothesis and again while building the test output.
- **With tools and no summary (S80):**
  - The model prints the grids first: turn 1 has a median of 1.5k tokens, about 1% of the attempt.
  - Before writing `solve()` it computes in Python:
    - shapes: 94% of attempts;
    - bbox / argwhere: 71%;
    - connected components: 56%;
    - colour counts: 48%;
    - in/out diff masks: 40%;
    - D8 checks: 12%.
  - That code is cheap, about 1.6k tokens per attempt.
  - It then reasons over the printed grids in prose with cell coordinates, which costs about 9.4k tokens (23% of reasoning characters).
  - Readers estimated perception at about 25% of S80 tokens (19% by hand, 6% in code).

**Does the summary simplify DeepSeek's thinking? Only a little. It replaces the cheap, table-like part of perception, but not the expensive cell-level part. The tokens it saves go mostly into longer prose hypothesis search.**

Paired V1 − S80 on the same 167 pairs (95% CIs from a task bootstrap):

| Measure | S80 | V1 | Change [95% CI] | Verdict |
|---|---|---|---|---|
| Output tokens | 40.7k | 37.1k | −3.6k [−6.1, −1.1] (−9%) | **down** |
| Verbatim digit-row lines | 61 | 45 | −16 [−25, −9] | **down** |
| Cell-coordinate reasoning tokens | 9.4k | 9.6k | +0.06k [−0.9, +0.9] | unchanged |
| Perception-only code cells | 8.3 | 6.8 | −1.5 [−2.1, −0.8] | **down**, but only about −0.2k tokens |
| Colour counts in code before `solve()` | 48% | 38% | −10 pp | **down** |
| Connected components in code | 72% | 78% | +6 pp | **up** |
| Diff masks in code | 55% | 51% | no change | unchanged |
| Turn 1, median | 1.5k | 2.4k | longer | **up** |
| Attempts with turn 1 ≥ 5k tokens | 18% | 36% | doubled | **up** |
| Tokens before the first `solve()`, median | 22.6k | 18.9k | mean −1.3k [−3.9, +1.3] | not significant |

- **Verbatim rows:** this is the E017 effect. Such rows are only about 0.3–0.4k tokens per attempt in tool runs.
- **Cell-coordinate reasoning:** about 9.5k tokens in both conditions.
- **Re-derivation:** the model **re-derives summary items in all 10/10 annotated V1 traces** (4 fully, 6 partly). The main reason is need, not distrust:
  - `solve()` needs per-cell masks, while the summary gives object-level numbers;
  - truncation ("12 largest shown", "+6 more") hid the decisive small objects (e3721c99, 446ef5d2);
  - the rule depended on edge cells or wire crossings that object tables cannot express (78332cb0, faa9f03d).
- **Turn 1:** the summary invites the model to theorise from it before running any code.
- **Hypothesis share:** annotated token shares moved from perception to hypothesis prose (S80 → V1):
  - hand perception: 19% → 15%;
  - hypothesis prose: 49% → 56%;
  - code: 18% → 13%.
- **Test audit:** the "Outside train range" line did not raise the test-vs-train audit rate (40% → 37% by regex). Readers saw it ignored in most traces.

**Does it ever trust a wrong summary item?**
- No numeric item was wrong in the 10 traces. The summary is deterministic.
- Its *framings* misled the model in 6/10 traces:
  - bg = the separator colour (271d71e2, 7491f3cf);
  - 8-connected multi-colour blobs gluing crossing wires (faa9f03d);
  - an object-level view that invited wrong hypotheses: shape pairing (58f5dbd5), sorting by size (78332cb0), the legend read as "bar + block" pairs (e87109e9, which then ran out at 80k with no answer), and a spurious `moved #5` match that it took at face value for a while (271d71e2).
- The reverse also happens: it distrusts correct items. For example it switched to 4-connectivity "out of doubt" (446ef5d2), and it blamed the summary for its own miscount.
- **Where the summary helped (5/10):**
  - it exposed per-object rotations (6ffbe589);
  - the L/dot split identified the marker (7491f3cf);
  - `holes=` plus "recoloured" or "removed" gave the rule at 3–20% (aa4ec2a5, e3721c99);
  - colour totals framed the jigsaw area (446ef5d2).

**DeepSeek's overall strategy** (both settings; details in §1):
1. Skim and guess roles.
2. Perceive per example: sizes, changed cells and objects as coordinates.
3. Propose a rule early. The core idea appears at a median **13–19%** of the trace in every condition.
4. Refine it against each example. The final rule settles late, at about 80–90%, and this phase is the largest token sink.
5. Apply to the test:
   - without tools, row by row by hand (about 21% of NT tokens), where most NT failures are row or count slips;
   - with tools, via `solve(test_input)`.

It handles uncertainty mostly in prose:
- It hedges a lot: "maybe/wait/hmm" appear about 20 times per 10k characters in NT and about 33 in tool runs.
- It patches contradictions or explains them away.
- When the budget runs low it ends with "given the time…" (17% of NT traces; 54% of tool traces, where the budget notes prompt it).

**For distillation into Qwen:**
- The no-summary tool traces already contain the perception the summary provides, done by code. That is the cheap and correct part to imitate.
- The cost worth pruning is cell-coordinate prose (about 9–10k tokens per tool attempt, about 21k without tools). The summary does not remove it.
- If V1-style traces are used for training, Qwen must also get the summary at inference. Those traces cite it constantly (about 23 summary-quoting lines per attempt, against 2.7 in S80). They also inherit its framing errors.

---

## 1. Overall strategy, phase by phase

### 1a. Token shares in the annotated sample
Readers' token-weighted estimates over 10 tasks × 3 conditions.

| Phase | NT (69k mean) | S80 (45k) | V1 (43k) |
|---|---|---|---|
| Perception by hand (transcribing, counting, listing coordinates) | **33** | 19 | 15 |
| Perception in code | 0 | 6 | 6 |
| Hypothesis search in prose | 33 | **49** | **56** |
| Writing and debugging code | — | 18 | 13 |
| Verification on train | 8 | 3 | 3 |
| Test application (building the output) | **21** | 4 | 6 |
| Second-guessing after the answer, or other | 5 | 2 | 2 |

Positions within the trace:

| | NT | S80 | V1 |
|---|---|---|---|
| Core idea first appears (median) | 18% | 19% | 13% |
| Final rule settled (median) | 81% | 92% | 82% |

These shares agree with E017 phase 1 on 48 S80/S96 traces: perception 20%, hypothesis 51%, code 11%.

### 1b. What the phases look like
**No tools (NT).** Excerpts are verbatim from `ds_annotations/` unless marked.
- **Opening skim:**
  > "We need solve ARC-like puzzle… output grid same size with some 4s changed to 8" (1818057f)
- **Per-example parsing**, as copied rows and indexed cells:
  > "r5: [4,2,2,2,2,2,4,2,8,8,8,2] changes c8,c9,c10… Coordinates (4,9),(5,8),(5,9),(5,10),(6,9). That's a plus shape"
- **Hypothesis, then a check on every example by hand:**
  > "But the number of 5s converted equals shift distance in all examples?" (271d71e2)
- **Building the test output by hand**, with recounts:
  > "Before final, let's double-check row count and all lengths. I'll annotate each row length." (aa4ec2a5)

  Most NT failures in the sample are hand-perception slips while doing this, not missing ideas:
  - 29 rows instead of 30 (e87109e9);
  - "10x11" for an 11×11 grid (78332cb0);
  - an area total of 217 instead of 223, never recounted (446ef5d2);
  - a 4-connectivity hole miscount (e3721c99);
  - 10 rows copied instead of 12 (faa9f03d).

  This matches E007, where 32% of wrong answers were execution errors.
- **Uncertainty:**
  - **contradictions explained away:** "mistake in the puzzle", or keys invented (e3721c99);
  - **"Given the time, I'll provide… my best guess"** (446ef5d2).

**With tools (S80).**
- **Turn 1 is short:** "Let me use code to inspect." Then it prints dims, diffs and grids.
  > `diff=np.argwhere(inp!=out); print("ex",i,"shape",inp.shape,"num diff",len(diff))` (1818057f, turn 1)
- **Long prose turns follow,** reading the printed output cell by cell:
  > "Row 8: [4, 1, 1, 4, 4, 6, 6, 4, 4, 8, 8, 8, 4, ...]. indices 0..: 0:4,1:1,2:1,3:4…" (58f5dbd5)
- **Then `solve()`, run against train, then submit.**
- **Good traces defer ambiguity to code:**
  > "maybe there's a subtlety. Let me just analyze via code." (aa4ec2a5)
- **Bad traces stay in prose and submit under budget pressure:**
  > "OK given time, maybe accept rule … and also incorporate a special case to make band1 work" (271d71e2)

  > "Final decision: submit the test input itself (identity)." (446ef5d2)

### 1c. Population-level signatures (regex over all attempts)

| | NT | S80 | V1 | V3 |
|---|---|---|---|---|
| Verbatim digit-row lines per attempt | 119 | 61 | 45 | 53 |
| Attempts with ≥40 such lines | 84% | 41% | 34% | 40% |
| Cell-coordinate lines (≥2 `(r,c)` tuples or `r5c3`/`c8=4` refs), not summary quotes | 225 | 110 | 113 | 135 |
| Est. tokens in those lines (chars ÷ chars-per-token) | 21.1k | 9.4k | 9.6k | 10.6k |
| Est. tokens in verbatim digit rows | 1.9k | 0.4k | 0.3k | 0.3k |
| Hand transcription share of reasoning characters | 40% | 23% | 25% | 27% |
| Lines quoting summary or tool formats (`bbox=`, `#3 c4`, `shape=G`…) | 1.4 | 2.7 | **23.3** | 21.1 |
| "maybe / wait / hmm…" per 10k characters | 19.6 | 33.5 | 31.6 | 32.0 |
| "given the time / best guess / go with this" (any) | 17% | 54% | 55% | 53% |
| Test-vs-train audit (E017 regex) | 39% | 40% | 37% | 47% |

- **The coordinate-line metric mixes two things:** perception, and mental execution of geometry such as reflections, ray paths and port directions. I checked 24 random lines; it is "thinking in cell coordinates", which the summary does not remove.
- **Wrong attempts carry 2–3× more of it:**

  | | NT | S80 | V1 |
  |---|---|---|---|
  | Correct attempts | 15.3k | 6.8k | 5.9k |
  | Wrong attempts | 31.4k | 15.6k | 17.8k |

  This is a correlation; harder tasks cause both.

## 2. Self-computed summary statistics

### 2a. Per statistic
From the 10 annotated tasks: how each item was obtained. Numbers count traces; "both" means by hand and in code.

| Summary item | NT | S80 | V1 |
|---|---|---|---|
| Grid dims | hand 10 | code 6, both 3, hand 1 | summary 10 |
| Background | hand 10 | hand 5, code/both 4 | summary 8 |
| Colour counts | hand 6, none 4 | code/both 5, none 4 | summary 5, hand/both 3 |
| In/out cell diff | hand 8 | code/both 5, hand 3 | summary 3, code 3, hand 2 |
| Objects (bbox, size) | hand 10 | code/both 8, hand 2 | **both 6**, summary 3, code 1 |
| Holes (relevant in 4 tasks) | hand 4 | code/both 2, hand 1 | summary or code |
| D8 / symmetry | hand 5 | code 2, hand 3 | code 3, summary 2 |
| Crop / scale / tile / panels | hand 6 | code/both 5 | both 4, code 1 |
| In→out object matching | hand 10 | hand 5, code/both 5 | summary 4, both 3, hand 2 |
| Test vs train range | hand 6, none 4 | hand 5, code 3 | summary 5, code/hand 4 |

### 2b. In Python, over all tool attempts
"Pre" means in cells before the first `solve`/`transform` definition; "any" means anywhere.

| Code pattern | S80 pre / any | V1 pre / any | V3 pre / any |
|---|---|---|---|
| `.shape` / len | 94 / 99% | 81 / 96% | 83 / 98% |
| `np.unique` / Counter | 48 / 61% | 38 / 57% | 45 / 59% |
| Components (BFS/label) | 56 / 72% | 55 / **78%** | 55 / 77% |
| bbox / argwhere | 71 / 87% | 64 / 84% | 65 / 89% |
| in≠out diff | 40 / 55% | 37 / 51% | 28 / 53% |
| D8 (rot90/flip/.T) | 12 / 21% | 10 / 19% | 19 / 23% |
| Print grids | 52 / 69% | 40 / 65% | 48 / 70% |
| Perception-only cells per attempt | 8.3 | 6.8 | 6.8 |
| Their tokens (chars ÷ 3) | 1.63k | 1.45k | 1.43k |

**Reading:** the summary trims the *explicit perception cells* by about 1.5 per attempt: fewer `np.unique` and grid prints. Component code becomes, if anything, *more* common, because the solver needs masks. In V1 the model often checks its own code against the summary:
> "Hole counts match summary" (e3721c99)

> "Let me replicate: objects = 8-connected cells of one colour." (V1, population grep)

### 2c. How V1 relates to the summary (all 167 V1 attempts, regex)

| Pattern | Attempts |
|---|---|
| "summary says / shows / lists" | 146/167 (87%) |
| "from / per / according to the summary" | 81 (49%) |
| "not rely / not trust / not reliable" | 33 (20%) |
| "summary … wrong / misleading / confusing" | 5 |

The "not rely / not trust" phrases are often about its *own* readings, not the summary. Typical usage is as a lookup table that corrects hand slips:
> "from summary: #3 c1 size=1 bbox=(4,15)-(4,15)… I mislabeled column. Let me use bbox." (16de56c4)

Typical distrust of a *correct* item:
> "summary says size 16. So maybe some cells differ… Let me trust code later." (the model's own hand count was off)

> "I think the summary is auto-generated with possibly 4-connectivity… Not reliable. Let me go back to basics." (it is 8-connected, as stated)

The connectivity convention (8-connected single-colour objects alongside multi-colour objects) is the most common source of confusion.

## 3. Paired comparison, S80 against V1 (same 167 pairs)
S80 is averaged over its 3 samples per pair. V1 has one sample. CIs come from a task-cluster bootstrap with 5k resamples.

| Metric | S80 | V1 | Δ [95% CI] |
|---|---|---|---|
| Output tokens per attempt | 40.7k | 37.1k | **−3.6k [−6.1, −1.1]** |
| Turn-1 tokens, mean (median) | 3.1k (1.5k) | 5.3k (2.4k) | **+2.3k [+1.2, +3.4]** |
| Attempts with turn 1 ≥ 5k tokens | 18% | 36% | — |
| Tokens before the first `solve()` (median) | 26.0k (22.6k) | 24.7k (18.9k) | −1.3k [−3.9, +1.3] |
| Cells before the first `solve()` | 6.7 | 5.4 | −1.3 [−2.2, −0.5] |
| Verbatim digit-row lines | 61.3 | 44.9 | **−16.4 [−24.6, −8.9]** |
| …of which in turn 1 / in later turns | 7.7 / 53.6 | 10.6 / 34.3 | — |
| Own cell-coordinate lines | 109.8 | 112.6 | +2.8 [−8.4, +13.8] |
| Own transcription tokens (rows + coordinate lines) | 9.8k | 9.8k | +0.06k [−0.9, +0.9] |
| Summary or tool quoting lines | 2.7 | 23.3 | +20.6 |
| Perception-only code cells | 8.3 | 6.8 | **−1.5 [−2.1, −0.8]** |
| Colour counts in code before `solve()` | 48% | 38% | −10 pp [−17, −3] |
| Components in code (any) | 72% | 78% | +6 pp [+1, +11] |
| Test-vs-train audit | 40% | 37% | −3 pp [−10, +4] |

**Other comparisons:**
- **V1 against S96** (no summary, 96k budget): −8.6k tokens and −1.6k own-transcription tokens [−2.5, −0.6]. Part of that gap is the budget, not the summary.
- **V3 against S80:** −1.5k tokens (not significant), longer turn 1 (+4.1k), and +2.1k transcription tokens. The ROLES prompt adds prose.

**Where the −3.6k tokens come from:**
- fewer copied rows, about −0.1k;
- fewer perception cells, about −0.2k;
- fewer turns and cells, plus a shorter tail. Correct V1 rollouts have p50 22.5k against 25.4k for S80 (E017).

Not from a smaller perception phase. Turn 1 grows because the model now theorises from the summary before it runs code. In the annotated sample, the time to the core idea dropped slightly (median 19% → 13%), and in 3 tasks the rule came almost straight from the summary:
- aa4ec2a5, at 3%;
- e3721c99, at about 20%;
- 6ffbe589, at 17% against 79% for S80.

In others the summary's object framing *delayed* it: 58f5dbd5 at 88% against 14%, and e87109e9.

## 4. Caveats
- Regex metrics are proxies. "Cell-coordinate lines" mix perception with mental simulation. Characters are converted to tokens with per-attempt reasoning chars per token: about 3.0 in tool runs, 2.5 in NT.
- NT is max effort with a different prompt and no budget notes, so NT-vs-tool differences are not only about tools.
- **Annotation sample:**
  - 10 tasks, chosen on purpose to mix outcomes; one reader per trace.
  - Shares are estimates.
  - Reader D's excerpts are lightly abridged.
  - A content filter stopped several readers; 4c3d4a41 has only its NT annotation.
- V1 is a single sample, so per-pair noise is large. The paired CIs account for task clustering but not for sample-to-sample variance within V1.

## Files
- **Scripts** (`scripts/`; run with `.venv/bin/python -I`; no API calls):

  | Script | What it does |
  |---|---|
  | `ds_features.py` | Per-attempt features → `ds_features.jsonl` (1,527 rows) |
  | `ds_stats.py` | Every table above |
  | `ds_render.py` | Renders NT / S80 / V1 traces for reading |
  | `ds_annot_agg.py` | Aggregates the annotations |

- **Annotations:** `ds_annotations/{A..F}.jsonl` (31 traces plus COMPARE lines) and `ds_annotations/annotation_guide.md`.
