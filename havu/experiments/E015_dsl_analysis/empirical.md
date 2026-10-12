# E015 empirical: how useful is Hodel's arc-dsl for ARC-AGI-2?

No API calls and no GPU were used. CPU time was about 45 min (most of it the search, run on 20 workers).

- **arc-dsl version:** `vendor/arc-dsl` @ `635de4902a5fb4e376f27333feaa396d3f5dfdcb` (2024-10-11). It was read before running; `dsl.py` and `solvers.py` import only each other, `arc_types` and `constants`, with no I/O.
- **ARC-AGI-1 data:** `fchollet/ARC-AGI` @ `399030444e0a`, from the local clone `/home/vu/Workspace/auto-neurogolf/ARC-AGI/data`.
- **ARC-AGI-2 data:** `data/arc-agi_training_*` (1000 tasks) and `data/arc-agi-2-github/data/evaluation` (120 tasks, 167 test pairs).

## 1. Inventory (`inventory.py` → `inventory.json`)

**Primitives: 160.** All 160 are used by at least one solver.

| Category | Primitives | Uses across the 400 solvers |
|---|---|---|
| Object extraction and properties (objects, partition, ofcolor, corners, size, ...) | 29 | 848 |
| Set and container ops (sfilter, argmax, merge, order, ...) | 28 | 889 |
| Grid transforms (rotations and mirrors, up/downscale, concat, split, crop, compress, period) | 28 | 531 |
| Arithmetic and logic | 24 | 570 |
| Functional combinators (compose, chain, fork, rbind/lbind, apply/mapply, branch, power) | 14 | **952** |
| Patch geometry (shift, normalize, box/inbox/outbox, backdrop/delta, connect, shoot, frontier) | 11 | 313 |
| Colour (mostcolor, palette, recolor, replace, switch, ...) | 10 | 369 |
| Spatial relations (adjacent, manhattan, gravitate, neighbors, ...) | 10 | 97 |
| Painting (fill, paint, underfill, underpaint, cover, move) | 6 | 344 |

The DSL has no loops beyond `power(f, n)` and `apply`. It also has no recursion, no mutable state and no lookup tables.

**Solvers: 400**, one per ARC-AGI-1 training task. Each is straight-line: one call per line.

| Stat | Mean | Median | p90 | p95 | Max |
|---|---|---|---|---|---|
| Calls (= lines) | 12.4 | 9 | 25 | 34 | 60 (97a05b5b) |
| Distinct primitives | 12.3 | 10 | 23 | 30 | 45 |
| Combinator calls | 4.2 | 2 | 10 | 17 | 45 |

- **Solver length distribution:** 40 solvers have 1–3 calls, 95 have 4–7, 161 have 8–15, 77 have 16–30, and 27 have more than 30. The average solver is about 375 characters.
- **Most-used primitives:** objects (217 solvers), fill (166), mapply (139), compose (131), ofcolor (124), rbind (123), apply (118), fork (114), paint (112), lbind (108), shift (103).

**Sanity check:** with `python -I`, 399 of the 400 solvers reproduce every pair of their ARC-AGI-1 task. The exception is e40b9e2f, which gets 2 of 4 pairs; the failure looks dependent on the hash seed.

## 2. Carry-over from ARC-AGI-1 to ARC-AGI-2 (`carryover.py` → `carryover.json`)

**ARC-AGI-2 training (1000 tasks)**

| Source | Tasks |
|---|---|
| ARC-AGI-1 training IDs | 391 |
| ARC-AGI-1 eval IDs | 376 |
| New IDs | 233 |

- **The reused tasks are unchanged.** All 391 training-ID tasks, and 375 of the 376 eval-ID tasks, contain exactly the same pairs; only the order of the train pairs differs. The train/test split is the same.
- **Hodel solvers on the ARC-AGI-2 copies:** 391 tasks have a solver, and 390 pass every pair (99.7%; pair accuracy 99.9%). The one failure is e40b9e2f, which already fails on ARC-AGI-1. The carry-over is therefore trivially 100% of working solvers: these are the same tasks.
- **Coverage:** Hodel's solvers cover 39% of ARC-AGI-2 training and none of the 233 new tasks.
- **The 233 new tasks are not reskins:** none has at least 50% of its grids matching a single ARC-AGI-1 task (D8 and colour-permutation canonical hashes from `data_tools/contamination_check.py`).

**ARC-AGI-2 evaluation (120 tasks)**

- **No overlap with ARC-AGI-1 training,** by ID or by content.
- **6 tasks are verbatim ARC-AGI-1 evaluation tasks** with the same ID and the same pairs, reordered: 0934a4d8, 136b0064, 16b78196, 981571dc, aa4ec2a5, da515329. Hodel has no solvers for ARC-AGI-1 evaluation tasks.
- **2 more tasks share grids with ARC-AGI-1 eval:**
  - db695cfb: 10 of its 12 grids match ARC-1 eval 55783887, so it is a near-duplicate.
  - 89565ca0: 2 of 7 grids match, spread across tasks.
- **Total: 8 of 120 tasks** share at least one grid with ARC-AGI-1, and 7 are at least 50% one ARC-1 task.
- **Contamination note:** these 6–7 tasks are in the public ARC-AGI-1 eval set and in ARC-AGI-2 training (the 376 ARC-1 eval IDs include them). Anything trained on ARC-AGI-2 training has seen them. da515329, the spiral task, is still unsolved by every config we ran.

## 3. Expressiveness on ARC-AGI-2 eval (`extract_programs.py`, `review/*.jsonl`, `dsl_solvers.py`)

**How the programs were extracted**

- **Source runs:** the 10 `tools_persistent_grid_lean2sub*_full_*` runs (E009, E010 and E011) contain 1955 correct attempts covering 111 distinct eval tasks.
- **What counts as the program:** the `submit_answer` expression plus the closure of the session's top-level `def`s it uses. When the submitted expression is just `answer`, the statements that last assigned it are used instead. 110 of the 111 programs are function-based. e12f9a14's only correct answer was a hand-entered script.
- **One program per task:** the median-length correct program, stored in `programs/<task>.py`.
- **Size:** median 48 non-blank, non-import lines, mean 51, p90 80, max 122.
- **Known gap:** top-level constant dicts such as `rules` and `map_pos` are not captured.

**Who judged:** I judged every task myself (a single grader). The subagent reviewers were stopped by a rate limit, so I read every program and the grids where needed.

- **Solved sample:** 40 tasks, a seeded random sample of the 111 (`sample_solved.txt`).
- **Hard set:** 20 tasks (`sample_hard.txt`).
  - The 8 tasks that have a test pair no config has ever solved, across all 78 result files: 4c7dc4dd, 5545f144, 88bcf3b4, abc82100, da515329, faa9f03d, a32d8b75, 88e364bc.
  - The 12 with the lowest correct rate in the tool runs.
- **Rule source:** E007 `true_rule` where available.

**Verdict definitions**

- **yes:** a Hodel-style solver of about 40 lines or fewer, using existing primitives.
- **extra:** concise once 1–3 generic primitives are added.
- **no:** needs control flow, search or simulation that the DSL lacks.

### Verdicts

| Set | yes | extra | no |
|---|---|---|---|
| 40 solved eval tasks | **14 (35%)** | 15 (38%) | 11 (28%) |
| 20 hardest/unsolved tasks | 1 (5%) | 9 (45%) | 10 (50%) |

- **The "yes" verdicts were spot-checked.** I wrote real arc-dsl solvers for 6 of the 14 "yes" tasks (`dsl_solvers.py`): 1818057f, 58490d8a, 62593bfd, 8e5c0c38, aa4ec2a5, bf45cf4b. **All 6 pass every train and test pair.** They take 5–11 DSL lines, against 12–38 lines of DeepSeek numpy, and 2.4× fewer characters overall (2252 against 5304).
  - The one bug I made while writing them was mixing up the `Indices` and `Object` types (a cell set against (colour, cell) pairs). This is a real ergonomic trap of the DSL for an LLM.
- **Program length by verdict:**

  | Verdict | Median DeepSeek LOC | Estimated DSL lines |
  |---|---|---|
  | yes | 38 | ~20 |
  | extra | 56 | ~30 |
  | no | 55 | – |

- **Operations in the solved sample:**

  | Operation | Tasks (of 40) |
  |---|---|
  | Connected components | 23 |
  | Conditional rules | 15 |
  | Shape/template matching | 14 |
  | Symbol/legend lookup | 11 |
  | Grid partition by separators | 10 |
  | Counting | 10 |
  | Line drawing | 10 |
  | Arithmetic geometry | 9 |
  | Recolouring | 8 |
  | Crop | 8 |
  | Symmetry | 7 |
  | Gravity/movement | 6 |
  | Tiling | 5 |
  | Path tracing | 5 |
  | Sorting | 4 |
  | Contextual role binding | 4 |

- **What separates the verdicts:**
  - The "yes" tasks are object-centric ARC-1-style rules: components, recolour, move to an edge, fill holes, mirror.
  - The "no" tasks are dominated by path tracing, conditional rules, lookups and grid partitions.

### Missing primitives and concepts (solved sample + hard set)

| Missing | Solved (40) | Hard (20) | Examples |
|---|---|---|---|
| Control flow: path following with turns, step simulation, loop until fixpoint or border, graph or chain walks, fold with state, ray deflection, search/packing | 9 tasks | 8 tasks | 9bbf930d, cb2d8a2c, 78332cb0, d35bdbdc, 142ca369, e8686506, 5545f144, 2b83f449, da515329 |
| Lookup table / legend parsing (symbol → shape, colour or transform) | 7 | 4 | 136b0064, 65b59efc, eee78d87, abc82100, dbff022c |
| Split into a variable number of panels by separator lines | 4 | 1 | 135a2760, 38007db0, 65b59efc, 78332cb0 |
| Object orientation / pointing direction | 4 | 1 | 3dc255db, 409aa875, 64efde09, 80a900e0 |
| Ray or slide until an obstacle (`shoot` ignores obstacles; `gravitate` only targets an object) | 4 | 2 | 142ca369, 64efde09, 8b9c3697, 88e364bc |
| Fold/concat of a variable number of grids (stacking, bar charts) | 3 | 1 | 291dc1e1, 78332cb0, 89565ca0, 7b5033c1 |
| Choosing a D8 transform or shape equivalence up to rotation | 3 | 1 | 67e490f4, fc7cae8d, a32d8b75 |
| Gap detection on broken lines / noise-robust repair | 2 | 2 | 89565ca0, 135a2760, 7b80bb43, faa9f03d |

**Would the DSL help discovery on the hard set?** I judged it "partly" for 7 of 20 tasks and "no" for 13. Those 13 fail on concepts: role binding, multi-rule composition, topology, or temporal extrapolation. A library of primitives does not supply those. This matches E007: most failures are partial rules, not missing machinery.

## 4. Enumerative search (`search.py`, `solver_transfer.py`)

**Setup**

- **Search:** breadth-first, depth up to 3, over 270–310 grid→grid arc-dsl expressions, with colours bound from each task's palette. The expressions cover:
  - rotations and mirrors, plus their concatenations and underpaint symmetry completions;
  - up/downscale, trim, compress and halves;
  - cellwise overlays of halves;
  - replace and switch for every colour pair;
  - largest/smallest/most-colourful object crop, cover and fill;
  - delta and outbox fills, enclosed-background fill, frontiers, and connecting aligned same-colour cells.
- **Pruning:** observational-equivalence (duplicate intermediate train-input states are dropped).
- **Scoring:** a program counts only if it fits every train pair and then also gets the test pairs right.

**Results**

| Set | Budget | CPU | Fits train | Test correct |
|---|---|---|---|---|
| ARC-AGI-2 eval (120) | 10 s/task | 20.3 CPU-min | 0 | **0 (0%)** |
| ARC-AGI-1 eval (400), sanity reference | 3 s/task | 20.3 CPU-min | 12 | 11 (2.75%) |
| Hodel's 400 solvers applied as-is to ARC-AGI-2 eval | – | 4.5 CPU-min | 0 | 0 |

- **How deep it got:** depth 1 was exhausted on every ARC-AGI-2 task (median 6.8k evaluations per task). Only about 7% of depth 2 was reached, and depth 2 was completed for just 1 task (eee78d87, small grids).
- **ARC-AGI-1 hits** were simple tasks: symmetry tilings, overlays of halves, upscaling, compress, and the most-colourful-object crop.
- **ARC-AGI-2 eval has no task solvable by one or two whole-grid ARC-1-style transforms.** Exhausting depth 2 would take about 145 s per task (~5 CPU-h for the 120). Given that depth 1 gives 0 and ARC-1 gives 2.75%, the expected gain is about 0–1 task.

## Assessment

- **Helper library for the tool harness — most promising.**
  - The DSL is pure Python with no dependencies. Exposing it in the sandbox (or a numpy-native subset) gives the model tested, bug-free versions of the operations it re-implements in almost every program: BFS components (23 of 40), bounding boxes, `occurrences`, mirrors, `delta`/`box`, enclosed-region fill and `gravitate`.
  - It would cut code length roughly 2–3× on the ~35% "yes" tasks, and remove a class of off-by-one bugs.
  - It will not fix the main failure mode, wrong or partial rules (E007), and it covers few of the hard tasks.
  - It would need about 8 added primitives, matching the table above: `ray_until_obstacle`, `slide_until_contact(dir)`, `split_by_separators`, `object_direction`, `fold_concat`, `canonical_shape` (D8), `lookup/legend` helpers and `flood_fill(seed)`.
  - The Indices-versus-Object type trap needs a clear docstring.
- **Search — not worth it on its own.** 0/120 with a depth-3 time-boxed search, and 0/120 for the 400 existing solvers. ARC-AGI-2 rules are too compositional for blind enumeration over this vocabulary. It could still work as a cheap verifier or repair step around LLM-proposed DSL programs.
- **Data generation — useful mainly for ARC-1-style skills.**
  - re-arc-style generators exist only for the 400 ARC-1 training tasks, which are already 39% of ARC-AGI-2 training. NVARC already has 1.13M augmentations from them.
  - The 233 new ARC-AGI-2 training tasks and the eval-style concepts (path tracing, legends, multi-rule role binding) are not covered. New generators would need the missing primitives first.
- **Token compression — modest.** DSL solvers are about 2.4× shorter than DeepSeek's numpy for expressible tasks. Output tokens in our runs are dominated by reasoning, not code (E009/E010), so the end-to-end saving would be small unless the model is trained or prompted to write DSL directly. Teaching 160 primitives in-prompt costs about 3–5k input tokens.

## Files

- **Scripts:**
  - `common.py`: loaders and the explicit vendor import (run everything with `.venv/bin/python -I`).
  - `inventory.py`, `carryover.py`, `extract_programs.py`, `summarise_review.py`.
  - `dsl_solvers.py`, `search.py`, `solver_transfer.py`.
- **Outputs:**
  - `inventory.json`, `carryover.json`.
  - `programs/`, `programs.jsonl`, `sample_solved.txt`, `sample_hard.txt`.
  - `review/solved.jsonl`, `review/hard.jsonl`.
  - `search_arc2_eval.json`, `search_arc1_eval.json`, `solver_transfer.json`.
- **Caveats:**
  - The verdicts come from a single grader, and "extra" against "no" is a judgement call.
  - The automatic regex `features` in `programs.jsonl` are coarse. The manual `ops` in `review/` are the ones to use.
