# Hodel: ARC-DSL and RE-ARC

- **Author:** Michael Hodel. arc-dsl dates from March 2023; RE-ARC from April 2024.
- **Code:** https://github.com/michaelhodel/arc-dsl (MIT) and https://github.com/michaelhodel/re-arc (MIT). Both read at HEAD on 2026-10-11.
- **Paper:** RE-ARC is arXiv 2404.07353. The arc-dsl writeup is `arc_dsl_writeup.pdf` in the repo.

Source tags: [W] = writeup/paper, [C] = code, [O] = our analysis (see `../experiments/E015_dsl_analysis/survey.md`).

## TL;DR
- **arc-dsl:** a functional Python DSL with 160 primitives over 20 hashable types (Grid, Object, Indices, Patch…). Solvers cover all 400 ARC-1 training tasks in straight-line style, one DSL call per line. [W,C]
  - Writeup counts: 235 of 400 solvers use ≤10 calls; 78 use >20.
  - Our count: median 9 lines. [O]
- **RE-ARC:** 400 hand-written example generators (median 34 lines [O]) with `diff_lb`/`diff_ub` difficulty bounds. Hodel-DSL solvers act as verifiers. [C]
- **ARC-AGI-2 relevance:**
  - 391 of the 400 tasks are in ARC-AGI-2 training; 0 are in eval. [O]
  - **The 400 solvers fit all train pairs of 0 of the 120 ARC-AGI-2 eval tasks.** Only 3 tasks have a solver that fits even the first train pair. [O]

## Design [W]
- **No classes.** An object is a frozenset of (colour, (i, j)) cells.
- **Primitives:** roughly a third each of transformations, properties and utilities. The utilities include `compose`, `rbind`, `branch`, `mfilter` and `argmax`.
- **Objectness:** `objects(grid, univalued, diagonal, without_bg)` gives 8 notions of object, plus 2 colour-partition variants. It appears in more than 50% of solvers.
- **Method:** iterate between writing solvers and refactoring the DSL. Remove rare primitives and promote frequent compositions.
- **"Shorter is better":** Hodel reports that train-fitting programs almost always also pass the test pairs on ARC-1.
- **Stated limitations:**
  - recursion and accumulation need "vast branching";
  - grid operations don't apply to patches;
  - direction and axis arguments are handled inconsistently;
  - the type set is redundant.

## Users
- **CodeIt** (Butt et al. 2024, arXiv 2402.04858): CodeT5 writes Hodel-DSL programs, trained with hindsight relabelling.
  - Result: 59/400 on the ARC-1 eval (pass@3).
  - It solves almost every task whose solution is ≤10 lines and none with 16+ lines.
- **RE-ARC data** was used by:
  - the ARChitects 2024;
  - Akyürek et al.'s TTT;
  - BARC Potpourri (100k transduction examples);
  - NVARC (102k samples, 3.2% of its mix).

## Reusable for us
- **Helper library:** `objects()` semantics and the property primitives are a good spec for a sandbox helper library (E015 option (a)).
- **Generators:** RE-ARC generators give verified primitive-skill data with a difficulty knob, but only at ARC-1 level. Use them as curriculum filler, not as ARC-AGI-2 coverage.
- **Program library:** don't use the solvers as a retrieval library for ARC-AGI-2 eval. Our check found 0 transfer.
