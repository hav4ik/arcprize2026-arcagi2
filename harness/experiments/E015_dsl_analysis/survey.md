# E015 survey: domain-specific languages (DSLs) for ARC, and what they are worth on ARC-AGI-2

Written 2026-10-11. This is desk research plus a few CPU-only measurements. It made no paid API calls and used no GPU. The empirical side of E015 (program inventory, search) is in `empirical.md` and the scripts in this folder, written separately.

Source tags:
- [P] paper or technical report;
- [R] repository code or README, read at HEAD on 2026-10-11;
- [B] blog or write-up;
- [O] our own measurement, with the script under `/tmp/e015scripts/` reproduced inline below.

Per-method summaries are in `prior_works/`:
- `hodel_arc_dsl_re_arc.md`
- `icecuber_dsl_search.md`
- `barc_induction_transduction.md`
- `dsl_systems_arc_agi_2.md`

## Bottom line
1. **No DSL-based system has shown a meaningful score on ARC-AGI-2.**
   - icecuber's 2020 DSL search scores **1.6% semi-private**, down from 17.0% on ARC-AGI-1. [P: 2505.11831 Table 1]
   - The best DSL-based ARC-AGI-2 results are LLM + DSL hybrids at 10–24% on the public eval. None is verified by ARC Prize, and all sit far below what our Flash + Python tool harness reaches (pass@1 ≈ 68%).
2. **Hodel's 400 ARC-1 solvers transfer to 0 of the 120 ARC-AGI-2 eval tasks.** None fits all train pairs. Only 3 tasks have a solver that fits even the first train pair. [O]
3. **Python with a helper library beats a restricted DSL whenever an LLM writes the code.** Every strong 2024–26 LLM program-synthesis system writes plain Python:
   - Greenblatt, Berman 2024, Pang, Imbue, SOAR;
   - BARC, which uses Python with a helper library;
   - NVARC's generators.

   BARC argues for this explicitly ("domain-specific libraries, not languages"). On ARC-AGI-2, Berman went further and dropped Python for English.
4. **Where DSLs have actually paid off since 2024 is data generation**, through RE-ARC, BARC, ARC-GEN and NVARC. Even there, the winners moved from a strict DSL (RE-ARC) to Python with helpers (BARC, NVARC).
5. **For us:**
   - (c) data generation is the one clearly useful role.
   - (a) a small helper library in the sandbox is cheap and worth an A/B test, mainly for Qwen.
   - (b) a DSL as the output language is not recommended.
   - (d) "token compression" has a low ceiling. Code is only 7% of Flash's output characters and 15% of Qwen's. [O]

## 1. The ARC DSL landscape

### 1.1 Hodel's arc-dsl and RE-ARC (michaelhodel)
**arc-dsl** (March 2023, MIT). [R, writeup `arc_dsl_writeup.pdf`]
- **What it is:** a purely functional Python DSL.
  - **160 primitives:** 79 unary, 67 binary, 13 ternary and 1 four-argument (`objects`).
  - **Types:** 20 types built from hashable tuples and frozensets: Grid, Object, Indices, Patch, Piece and so on. There are no floats, no dicts and no classes.
  - **Categories:** roughly a third each of transformations (mirror, rotate, recolor, move, fill…), properties (size, shape, bounding corners, centre of mass, `bordering`…) and utilities (set operations, arithmetic, `compose`, `rbind`/`lbind`, `branch`, `mfilter`, `argmax`…).
- **Objectness:** a single `objects(grid, univalued, diagonal, without_bg)` primitive gives 2³ notions of object. Two colour-partition variants add 2 more. It is the 6th most-used primitive and appears in more than 50% of solvers.
- **Solvers:** one for each of the **400 ARC-1 training tasks**.
  - **Rules:** written in a strict straight-line style. Each line is one DSL call whose arguments are constants or earlier variables. There are no native loops or branches.
  - **Length (writeup):** 235 of 400 solvers use ≤10 calls, 79 use ≤5, 78 use >20 and only 3 use >50.
  - **Length (our count):** median 9 lines, mean 12.4. [O]
- **Design method:** iterate between writing solvers and refactoring the DSL.
  - Remove primitives that are rarely used.
  - Promote compositions that are used often.
  - Aim for "shorter is better", because short programs overfit less.
  - Hodel reports that train-fitting programs almost always also passed the test pairs on ARC-1.
- **Self-stated limitations:**
  - Recursion and operator accumulation with depth unknown in advance can only be written as "vast branching".
  - Grid operations such as rotation, trimming and splitting don't work on patches.
  - Direction and axis arguments are handled inconsistently.
  - The type set is redundant (Indices vs Object).
- **Downstream users:**
  - **CodeIt** (Butt et al. 2024) trains CodeT5 to write Hodel-DSL programs, with hindsight relabelling: **59/400 = 14.75% on the ARC-1 eval** (pass@3). It "almost always solves the task if the solution program is ≤10 lines", never solves ground-truth programs of 16+ lines, and never learns some primitives. [P: 2402.04858]
  - **RE-ARC** (below).
  - ARC Prize 2024 credits "improved DSLs, notably Hodel's" as the main pre-LLM progress. [P: 2412.04604]

**RE-ARC** (April 2024, MIT). [P: 2404.07353; R]
- **Contents:** 400 hand-written generators (`generate_<id>(diff_lb, diff_ub)`), one per ARC-1 training task, plus 400 verifiers. The verifiers are Hodel-DSL solvers; only generated examples that pass them are kept.
- **Difficulty:** a `unifint(diff_lb, diff_ub, range)` call samples sizes and counts, so difficulty can be dialled.
- **Generalisation:** the generators lift constraints the originals didn't need (grid size, colours, object counts).
- **Size:** generators have a median of 34 lines (max 100). [O] A pre-generated zip has 1,000 examples per task.
- **Who used it:** the ARChitects 2024, Akyürek et al.'s TTT, BARC's "Potpourri" set (100k transduction examples) and NVARC (102k samples, 3.2% of its mix). [P: 2411.02272, NVARC paper]
- **Coverage limit:** it covers ARC-1 training concepts only. 391 of those 400 tasks are in ARC-AGI-2 training; 0 are in ARC-AGI-2 eval. [O]

### 1.2 icecuber (Johan Wind), 2020 Kaggle winner [R: top-quarks/ARC-solution, `ARC-solution_documentation.pdf`]
- **DSL:** C++ with 142 unary grid→grid transforms built from 42 functions, such as `Cut`, `filterCol`, `colShape`, `composeGrowing`, `compress`, `rigid` and `pickMax`.
- **Search:** enumerates chains of up to depth 4, removing duplicates. A greedy "stacker" overlays candidate images to fit the train pairs.
- **Ensemble:** depth 3, depth 3 with diagonal-flip augmentation, then depth 4 until the 9 h / 16 GB budget runs out. Candidates are ranked by train pairs solved, then depth, then stack size.
- **Cost of depth:** depth 3 is about 20× cheaper than depth 4, and depth 2 is about 15× cheaper again while solving about 80% as many eval tasks.
- **Where the transforms came from:** hand-solving 100 training and **100 evaluation** tasks. CodeIt notes that icecuber found a DSL designed on training tasks alone did poorly.
- **Scores:**
  - ARC-1 private: 20.6% (LB 0.794 top-3 error).
  - ARC-1 public eval: 39%; semi-private: 17%. [B: ARC Prize leaderboard post]
  - **ARC-AGI-2 semi-private: 1.6%.** [P: 2505.11831]
- **2020 meta-analysis:** pooling all 2020 entries, all brute-force DSL search, solved 49% of the ARC-1 private set. The best single DSL-search Kaggle entry later reached 40% (alijs). [P: 2412.04604] ARC-AGI-2 was explicitly designed to remove such tasks (design goal 3, "less brute-forcible"). [P: 2505.11831]

### 1.3 Other DSL program spaces (ARC-1 era)
- **GridCoder (Ouellette, 2024).** Transformer-guided enumeration over a custom DSL, growing from 74 to 89 to 98 primitives.
  - DSL v1 can express only **29/400 (7.25%)** of ARC-1 eval tasks.
  - Search time per primitive grows sub-linearly as the DSL grows.
  - Out of distribution, GridCoder solves only 1 of 10 tasks.
  - The author names "very limited work on improving the DSL" as the main limitation. [P: 2411.17708]
  - ARC Prize 2024 cites it as the example of "LLM-guided discrete program search over a DSL". [P: 2412.04604]
- **ARGA / NSA (2022, 2025).** An object-graph DSL. NSA's transformer proposes primitives and a combinatorial search finds the parameters, with test-time fine-tuning on generated tasks. It reports ARC-1 results only. [P: 2501.04424]
- **ILP and abductive solvers (2024).** Small hand-made object DSLs used as background knowledge. Results are on ARC-1 subsets only. [P: 2405.06399, 2411.18158]
- **LLM multi-expert agents (Tan & Motani 2023).** GPT-4 is given grid/object/pixel abstraction spaces and helper functions, with environment feedback: 50/111 ARC-1 training tasks. [P: 2310.05146]

### 1.4 The 2024 "DSL-free" turn (LLM program sampling, and transduction for contrast)
- **Greenblatt (Jun 2024):** about 8k GPT-4o **plain Python** programs per task, then filter and vote: 43% ARC-1 semi-private. See `prior_works/greenblatt_draw_more_samples.md`.
- **Berman 2024:** evolution over Python with Sonnet 3.5: 53.6% ARC-1 semi-private.
- **BARC (Li, Ellis et al., Nov 2024)** [P: 2411.02272; R: xu3kev/BARC]
  - **Seeds:** 100, later 160, hand-written Python programs. Each has an NL description, a `transform_grid` and a `generate_input`, all using a shared **`common.py` library**: `flood_fill`, `draw_line`, `find_connected_components`, `detect_objects`, `blit_sprite`, `crop`, `bounding_box`, `detect_*_symmetry`, `random_sprite`, `collision`/`contact` and so on.
  - **The paper's position:** DSLs "restrict the class of allowed programs… We still allow arbitrary Python code, which helps cover the long tail", and "we advocate… domain-specific *libraries*".
  - **Data:** LLMs remix the seeds into ARC-Heavy (200k problems) and ARC-Potpourri (400k, including RE-ARC).
  - **Results** (ARC-1 public eval, pass@2):

    | Model | Score |
    |---|---|
    | Induction (Llama-3.1-8B, 20k samples) | 38.0% |
    | Transduction with TTT and reranking | 43.0% |
    | Ensemble | 56.75% |
    | Ensemble, with Akyürek's better TTT | 61.9% |

  - **Scaling:** performance saturates as the number of seeds grows but keeps rising with the amount of synthetic data.
  - **False positives:** about 9% of train-fitting programs fail the test pair.
  - **Division of labour:** induction wins on counting and on composing several concepts. Transduction wins on fuzzy perceptual concepts.
- **MindsAI and the ARChitects (2024): transduction with TTT, no programs.** 55.5% and 53.5% ARC-1 private. ARChitects 2024 scores **2.5%** on ARC-AGI-2 semi-private. [P: 2505.11831] ARC Prize 2024: "the best transduction-only and induction-only single submissions score around 40%; only an ensemble of both can compete." [P: 2412.04604]
- **SOAR (2025, paper award 2nd):** self-improving evolutionary Python synthesis "without requiring human-engineered DSLs". ARC-1 only. [P: 2601.10904]

### 1.5 ARC-AGI-2 era (2025–26)
- **NVARC (Kaggle 1st, 24.03% private)** [R: 1ytic/NVARC, `nvarc_2025.pdf`]
  - **No DSL.** gpt-oss-120b writes **free Python**:
    - an input-grid program, kept only if it yields 30 or more unique inputs passing LLM-written unit tests;
    - 20 output programs per input program, kept if they agree on the outputs.
  - **Seeds:** 3,268 summaries drawn from H-ARC and BARC descriptions plus the eval tasks, mixed pairwise into 266k. 103k puzzles survive the filters.
  - **BARC data** was tested and dropped.
  - **Contamination:** 54.9% of training samples come from the eval-seeded `nvarc_full` subset (see AGENTS.md).
  - **Inference** is purely transductive (Qwen3-4B with TTT). Programs are used only to make data.
- **Pang (26.0% semi-private) and Imbue (Gemini 3 Flash 34→61.4% public):** both write Python. Pang's library of programs from earlier tasks gives no measurable transfer. See the existing summaries.
- **Berman 2025 (29.4% semi-private)** switched from Python to English: "This strategy hits a wall with ARC v2. The transformations are often too complex to express elegantly in Python — they require nuanced pattern recognition and contextual understanding that would result in unwieldy, brittle code." [B] This argues *against* an even more restrictive DSL.
- **Compositional Neuro-Symbolic Reasoning (Das et al., Stanford, Apr 2026)** [P: 2604.02434]
  - **Setup:** a fixed DSL of **22 atomic "Unit Patterns"** over object scene graphs. o4-mini detects which patterns apply (JSON, repeated 5×, top 3 kept). Grok-4 then writes the output grid **transductively**, with these as hints and 3–10 self-consistency samples.
  - **Results** (ARC-AGI-2 public eval, pass@2):
    - 24.4%, against a Grok-4 baseline of 16.0%;
    - 30.8% when ensembled with Berman's ARC Lang solver.
  - **Ablation:**
    - removing the symbolic hints: 24.4 → 17.5 (−6.9 pp);
    - removing self-consistency: 24.4 → 20.5.
  - **Failures** cluster on tasks "beyond the current 22-pattern DSL".
  - **Caveat:** this is the only ARC-AGI-2 ablation we found of "DSL in the prompt" against no DSL. It is a weak-model regime, and the DSL is used as **hints**, not as an output language.
- **LLM + genetic search over a 41-op DSL (Sep 2026)** [P: 2609.27242]
  - **Setup:** a Kaggle-constrained design. A quantized Qwen3.5-4B writes programs in a fixed-width DSL (27 core plus 14 LLM-gated ops), and a GPU GA evolves them, about 1.6M programs per task.
  - **Results** (first 60 public-eval tasks):

    | Stage | Solved |
    |---|---|
    | LLM programs alone | 2/60 (3.3%) |
    | Plus GA | 6/60 (10.0%) |
    | GA without LLM seeding | 0 |

  - **The authors' own caveats:** "A language with 41 operations can express far fewer ideas than plain Python." Tasks fail because "an operation was missing". 20 of the 60 tasks were used for development.
- **ARCANA (Jul 2026)** [P: 2607.09059] claims 32.5% "semi-private" with a 48M-parameter DSL-program model (47 typed primitives) under Kaggle constraints. **Treat it as unverified:**
  - no ARC Prize verification;
  - the semi-private set isn't available to authors;
  - the paper has boilerplate and irrelevant citations.
- **Living survey (Mar 2026)** [P: 2603.13372] reports a "2.5–3× degradation from ARC-AGI-1 to ARC-AGI-2 regardless of paradigm". Its table quality is uneven, so use it only as a pointer.

## 2. Why ARC-AGI-2 is harder, and how program search scored

**Design changes** (Chollet et al. 2025, §7). [P: 2505.11831]
- Tasks are more novel. None overlaps with earlier patterns.
- Tasks carry more information: "larger grids, more objects per grid, and more concepts per task… more bits per task".
- Tasks test compositional generalisation in four forms:
  1. **Multi-rule composition:** several rules interact, e.g. crop to a frame, rescale the objects, then place them into matching holes (`898e7135`).
  2. **Multi-step composition:** step N depends on step N−1, e.g. iterative placement (`cbebaa4b`).
  3. **Contextual rule application:** a context cue gates the rule, e.g. the outline colour decides the stacking side (`b5ca7ac4`). The paper calls this "a form of control flow".
  4. **In-context symbol definition:** objects stand for something defined inside the task, e.g. the number of holes encodes a colour. This is "a major challenge for frontier AI systems".
- **Explicit anti-brute-force goal.** ARC-1 was diluted by the 49% of tasks that pooled 2020 DSL search could solve.
- **Human baseline:** every task was solved by at least 2 people; 66% of attempted test pairs were solved; average 2.7 min per task.

**Our grid statistics** (train inputs; `dsl.objects(univalued, 4-conn, no-bg)`). [O]

| Set | Tasks | Median area | Share ≥400 cells | Median / mean objects | Mean colours |
|---|---|---|---|---|---|
| ARC-1 training (inside v2 train) | 391 | 100 | 8% | 4 / 10.6 | 3.4 |
| v2 train, other tasks | 609 | 121 | 14% | 6 / 16.1 | 4.1 |
| **ARC-AGI-2 public eval** | 120 | **300** | **42%** | **10 / 26.6** | **5.2** |

Eval inputs have 3× the area and 2.5× the objects of ARC-1 training. Search cost rises with both: perception ambiguity grows, and so does the number of objects a filter or argmax must pick from.

**How DSL and program-search systems scored on ARC-AGI-2:**

| System | Type | ARC-AGI-1 | ARC-AGI-2 | Source |
|---|---|---|---|---|
| icecuber 2020 | Brute-force C++ DSL | 17.0% semi-priv | **1.6%** semi-priv | 2505.11831 |
| ARChitects 2024 | Transduction + TTT | 56.0% | 2.5% | 2505.11831 |
| o3 (medium) | LLM, no tools | 53% | 3.0% | 2505.11831 |
| Hodel's 400 solvers reused as a library | DSL programs | 400/400 ARC-1 train (by construction) | **0/120** public eval fit train | [O] |
| LLM (Qwen3.5-4B) + GA over 41-op DSL | DSL hybrid, Kaggle | — | 10.0% (6/60 public) | 2609.27242 |
| Unit-Pattern DSL hints + Grok-4 | DSL hints, transductive | — | 24.4% public (30.8% ensemble) | 2604.02434 |
| ARCANA | DSL + 48M model | — | 32.5% "semi-priv" (unverified) | 2607.09059 |
| Pang 2025 | Python | 77.1% | 26.0% semi-priv | blog |
| Berman 2025 | English | 79.6% | 29.4% semi-priv | blog |
| NVARC 2025 | Transduction, Python-generated data | — | 24.03% private (Kaggle) | 2601.10904 |
| Imbue + Gemini 3 Flash | Python evolution | — | 61.4% public | blog |
| **Ours: Flash + Python tool** (E009/E011) | Python tool, grid answer | — | pass@1 67.8–68.3%, pass@2 ≈ 78% public | experiments.tsv |
| **Ours: Qwen3.8-27B + tool** (E012) | Same, 35.6k tok | — | pass@1 25.8%, pass@2 34.6% public | experiments.tsv |

How the top teams in 2025 scored: Kaggle 1st–5th place (NVARC 24.0, ARChitects 16.5, MindsAI 12.6, Lonnie 6.7, Barbadillo 6.5) were all TTT/transduction-centred. None was DSL search. [P: 2601.10904]

## 3. Does giving an LLM a DSL help?

**For:**
- Hint-style DSL on a weak LLM: +6.9 pp on ARC-AGI-2 public (Grok-4: 17.5 → 24.4) in 2604.02434. This is the only ARC-AGI-2 ablation found, and it is a single unreplicated preprint.
- LLM seeding of a DSL GA: 0 → 6/60 (2609.27242). That shows the LLM matters, not the DSL.
- A DSL makes programs short and mutation-safe. It enables massive cheap search (1.6M programs per task on a GPU) and exact verification.
- CodeIt and GridCoder learn DSL use from synthetic programs and solve short programs reliably.
- In the ARC-1 era, a good DSL was the main driver of progress (icecuber, Hodel).

**Against:**
- **Coverage, i.e. missing primitives.** GridCoder DSL v1 covers 7% of ARC-1 eval. Hodel's DSL needed "vast branching" for recursion. Every ARC-AGI-2 DSL paper lists missing operations as the main failure. ARC-AGI-2's "contextual gating" and "in-context symbols" are exactly what fixed primitive sets express badly.
- **Short-program bias.** CodeIt never solves ground-truth programs of 16+ lines. ARC-AGI-2 tasks have "more bits per task", so the solutions are longer.
- **The strongest LLM synthesisers chose Python** (Greenblatt, Berman 2024, Pang, Imbue, SOAR, BARC, NVARC). BARC's stated reason is the long tail. Berman found even Python "unwieldy, brittle" for v2 and moved to English.
- **Library priors don't transfer to new tasks.** Pang's 538-program library: no eval task has a program fitting all train pairs. Hodel's 400 solvers: 0/120 [O]. Reusable *primitives* may help; reusable *programs* don't.
- **Our own data:**
  - E004: forcing a `transform()` submission (code mode) was worse than grid answers with a Python tool. The model loses its best-guess exit.
  - E007: most Flash failures are partial rules (45%) or execution errors (32%), not "no vocabulary for the idea".
  - Python already covers execution. A DSL fixes neither failure type.
- **No direct DSL-vs-Python ablation for a strong LLM on ARC exists.** The nearest evidence is BARC's argument and the Berman/Pang/Imbue design choices.

**Used mainly for data rather than inference?** Yes, since 2024.
- RE-ARC (strict DSL generators) fed TTT models.
- BARC moved to Python with a helper library.
- NVARC used free Python generators, with LLM-written unit tests and multi-sample agreement in place of DSL verifiers.
- ARC-GEN (Google) has 900 Python generators covering 891 of the 1,000 v2 training tasks. They use a `common` helper module and are the official generator for the 2025 Code Golf and 2026 NeuroGolf championships. [R]
- The only inference-time DSL systems on ARC-AGI-2 are the hybrids in §1.5. They score below plain frontier LLMs and well below our harness.

## 4. Assessment for our setting

Context:
- **Flash + Python tool:** pass@1 ≈ 68%, about 41k tokens per attempt. The contamination caveat on the public eval applies.
- **Kaggle:** Qwen3.8-27B on 4×L4 at about 35k tokens per attempt, pass@2 34.6% on the public eval with tools.
- **Plan:** SFT a small model on generated data.

**Measurements from our own traces.** [O]
- **Flash** (E009 sub80s40, 538 attempts; E004 grid/lean2/low, 675 attempts):
  - **tool-call code is only 7.1–7.3% of assistant output characters**; the rest is reasoning;
  - median about 7k code characters and 10–11 tool calls per attempt;
  - the model already writes `scipy.ndimage.label` in 49–53% of attempts, hand-written BFS/flood fill in 28–29%, bounding boxes in 50–54%, colour counts in 60–65%, symmetry and rotation checks in 22–26%, and defines a `transform`/`solve` in 70–73%.
- **Qwen** (E012 v3, 141 traces): code is **15.3%** of assistant characters.

**(a) An importable helper library in the sandbox. Worth a cheap A/B, mainly for Qwen.**
- **Contents:**
  - `objects(grid, univalued, diagonal, bg)` with Hodel's 8 object notions, returning records (cells, bounding box, colour, size, shape key, holes);
  - `bbox`/`crop`;
  - the 8 dihedral transforms and `symmetries(grid)`;
  - `diff(pred, expected)` and `check(transform)` over the train pairs;
  - colour histograms;
  - a "legend/key detector" (repeated small motifs, framed regions).
- Document it in ≤300 prompt tokens.
- **Expected gain:**
  - Flash already writes these routines quickly, and code is 7% of its output. Gains would come from fewer execution bugs (E007: 32% execution errors) and fewer exploratory tool turns, not from shorter code.
  - Qwen spends a larger share on code and has a tighter 35k budget, so it benefits more.
  - Bound the gain at a few percent of tokens plus whatever accuracy fewer bugs buys.
- **Risks:**
  - **Anchoring:** the model forces the task into library objects. This is the ARC-AGI-2 failure mode of fixating on surface patterns.
  - **Library bugs** become systematic errors.
  - **Prompt-length overhead** on every turn.
- **Test:** BARC-style "library, not language": optional imports only. Run on the E009 56-pair dev subset with a paired comparison (`evals/paired_vs_baseline.py`).

**(b) A DSL as the output language for program search and verification. Not recommended.**
- Every ARC-AGI-2 data point is negative:
  - icecuber 1.6%;
  - DSL + GA 10% on 60 tasks;
  - Hodel solver reuse 0/120;
  - Berman leaving even Python;
  - our E004 code mode trailing grid mode.
- A restricted language cannot express contextual gating or in-context symbols without growing into Python.
- Keep **Python as the program language with train-pair verification**, which is the shared core of every strong harness.
- **Narrow exception:** a CPU-only DSL search (icecuber binary, about 9 CPU-hours) could run alongside Qwen on Kaggle at almost no GPU cost. Its expected yield is about 1–2% of tasks, mostly ones Qwen already solves. Only worth it if a quick check shows it solves tasks Qwen misses. Low priority.

**(c) A data-generation engine for SFT. The clear win; use Python generators with a shared helper library, not a strict DSL.**
- **Why:** generators give unlimited verified tasks with ground-truth programs. That supports:
  - (task → program) SFT;
  - (program + diff → fixed program) repair SFT;
  - RL with an exact-match reward;
  - and, in our case, synthetic tool-use traces where the oracle program supplies the right `check()` result.

  BARC shows accuracy keeps rising with the amount of synthetic data while seed count saturates. NVARC shows that scale plus two-concept mixing won Kaggle.
- **Sources, in order:**
  1. ARC-GEN (891 v2 training tasks, Apache-2.0);
  2. NVARC `nvarc_training` (47k puzzles from v2 training parents only);
  3. RE-ARC (400 ARC-1 concepts with difficulty control; good for primitives, too easy for v2);
  4. BARC Heavy (ARC-1 flavour; NVARC found it unhelpful for its model).
- **Exclude:** `nvarc_full` and anything seeded from eval.
- **The gap to fill with our own generators:** the four ARC-AGI-2 composition types (legends and symbols, context-gated rules, multi-step dependent placement, 3+ interacting rules) on large cluttered grids. Use the NVARC recipe:
  - LLM-written input generators with unit tests;
  - several output programs that must agree;
  - Flash as the writer at off-peak prices;
  - the same helper library as in (a), so the SFT model learns to call it.
- **Risks:**
  - the generator distribution mirrors the author's concept prior (RE-ARC is ARC-1-shaped);
  - mixing two easy concepts doesn't produce ARC-AGI-2-style symbol semantics;
  - false-positive programs (BARC: 9%) and ambiguous examples;
  - contamination: dedup against eval with `data_tools/contamination_check.py`.

**(d) DSL as compression of reasoning (shorter programs → fewer tokens). Low ceiling at inference.**
- Code is 7% (Flash) or 15% (Qwen) of output characters. Even halving program length saves only about 3–8% of tokens.
- The cost is in deciding *what* to compute, i.e. the reasoning, not in typing it.
- Hodel's median 9-line programs are short, but Hodel reports some took "a very challenging and long process" to write. For an LLM, a terse DSL raises the cost of reasoning about its semantics.
- **The real compression lever is perception helpers** that replace pages of manual grid reading with one printed summary (object lists, diffs): option (a).
- A DSL could compress reasoning only for a model *trained* to think in it (CodeIt-style SFT). That is a long project with a short-program ceiling (CodeIt: ≤10-line programs).

**Overall risk given ARC-AGI-2's concepts:**
- DSLs encode an ARC-1 prior: objects, symmetry, gravity, recolouring.
- ARC-AGI-2's added difficulty is semantic: what a symbol means, which context selects which rule, and stateful multi-step construction. Fixed primitives handle that worst.
- Use DSL-like code as **scaffolding** (helpers, verifiers, generators), never as a **cage** (the only output language).

## 5. Suggested cheap follow-ups (no paid calls unless approved)
1. Write `arc_utils.py` for the sandbox (BARC `common.py` / Hodel `objects` semantics). Run a dev-subset A/B on Qwen first; it costs $0 API on a self-hosted box.
2. Generator pilot: 50 Flash-written ARC-AGI-2-style generators targeting the four composition types, using the NVARC filters. Estimate cost before running (off-peak; roughly 20 calls per generator).
3. Optional: run icecuber's binary on the public eval on CPU to measure overlap with Qwen's solved set. It costs CPU time only.

## Sources
- Chollet et al., *ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems* (2025): https://arxiv.org/abs/2505.11831
- *ARC Prize 2024: Technical Report*: https://arxiv.org/abs/2412.04604
- *ARC Prize 2025: Technical Report*: https://arxiv.org/abs/2601.10904; results blog: https://arcprize.org/blog/arc-prize-2025-results-analysis
- Hodel, arc-dsl and its writeup: https://github.com/michaelhodel/arc-dsl
- Hodel, RE-ARC: https://github.com/michaelhodel/re-arc, https://arxiv.org/abs/2404.07353
- Wind (icecuber), *DSL solution to the ARC challenge*: https://github.com/top-quarks/ARC-solution
- Li et al., *Combining Induction and Transduction for Abstract Reasoning* (BARC): https://arxiv.org/abs/2411.02272, https://github.com/xu3kev/BARC
- Butt et al., *CodeIt*: https://arxiv.org/abs/2402.04858
- Ouellette, *Towards Efficient Neurally-Guided Program Induction for ARC-AGI* (GridCoder): https://arxiv.org/abs/2411.17708
- *NSA: Neuro-symbolic ARC Challenge*: https://arxiv.org/abs/2501.04424
- Tan & Motani, *LLM as a System of Multiple Expert Agents for ARC*: https://arxiv.org/abs/2310.05146
- Das et al., *Compositional Neuro-Symbolic Reasoning*: https://arxiv.org/abs/2604.02434
- *Combining LLMs and Genetic Search for ARC-AGI-2*: https://arxiv.org/abs/2609.27242
- *ARCANA*: https://arxiv.org/abs/2607.09059
- *The ARC of Progress towards AGI: A Living Survey*: https://arxiv.org/abs/2603.13372
- NVARC repo and paper: https://github.com/1ytic/NVARC
- Berman, *How I got the highest score on ARC-AGI again swapping Python for English*: https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again
- Google ARC-GEN: https://github.com/google/ARC-GEN
- Existing summaries: `prior_works/{greenblatt_draw_more_samples,pang_library_evolution,imbue_arc_agi_2_evolution,berman_nl_instruction_evolution}.md`

## Reproducing the [O] numbers
Clone arc-dsl and re-arc into a scratch directory, then run with `.venv/bin/python -I`:
- **Solver lengths, ID overlap and grid statistics:** parse `solvers.py`; take `data/arc-agi_training_challenges.json` and the eval JSON; compute area, `dsl.objects(g, True, False, True)` and colour counts on the train inputs.
- **Solver transfer:**
  - run all 400 `solve_*` functions (0.5 s timeout per call) on every train pair of the 120 eval tasks and of the 609 v2 training tasks whose IDs are not ARC-1 training IDs;
  - a solver counts only if it fits all train pairs;
  - eval: 0/120 full fits, and 3 with a first-pair fit;
  - those 609 v2 training tasks: 12 full fits (e.g. `ea9794b1` ← `75b8110e`), mostly ARC-1 eval near-duplicates.
- **Trace code share:** parse `evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40/*.messages.json.gz` and `tools_persistent_grid_lean2_deepseek-flash_low_full_a0/`. Compare `tool_calls[].function.arguments.code` against `reasoning_content` + `content`, and regex for helper patterns. For Qwen, use `<parameter=code>` blocks in `qwen_tools_medium_48k_full_v3_traces/*.txt.gz`. The figures are characters, not tokens; code tokenises slightly denser, so its token share is somewhat higher.
