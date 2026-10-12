# Training-data contamination check against the ARC-AGI-2 eval set (2026-10-02)

- **Script:** `contamination_check.py`. Log: `contamination_check.log`. Per-task counts: `contamination_report.json`.
- **Method:**
  - Every grid is canonicalised so that rotations, reflections and colour relabelling don't hide a match.
  - The result is compared against all 1,041 distinct non-trivial grids of the 120 eval tasks: train inputs/outputs, test inputs, and **test outputs (the answers)**.
  - Grids with fewer than 16 cells or a single colour are ignored.

## Results

| Source (intended clean) | Grids checked | Matches | Eval tasks hit |
|---|---|---|---|
| ARC-AGI-2 training (official) | 6.6k | 1 | 1 (`89565ca0`) |
| NVARC synthetic `nvarc_training` | 2.55M | 452 | 10 |
| NVARC augmented `arc2_training` | 1.21M | 217 | 3 |
| NVARC augmented `rearc` | 1.20M | 93 | 7 |
| NVARC augmented `concept` | 0.39M | 0 | 0 |
| NVARC augmented `mini` | 0.31M | 840 | 2 |
| NVARC augmented `nvarc_training` | 11.2M | 2,272 | 13 |

**Where the matches are:**
- Almost all fall on a handful of small grids in five tasks: `f931b4a8`, `20270e3b`, `2d0172a1`, `89565ca0` and `eee78d87`. These are mostly 4×4 to 6×6 training outputs.
- The 840 Mini-ARC matches are all one 5×5 output of `2d0172a1`. Mini-ARC consists entirely of 5×5 grids.
- A separate check of the eval **test outputs** alone found a single match in both the official training set and NVARC synthetic: the 5×6 answer of `89565ca0`.

## Conclusion
- **No meaningful leakage.** The matches are small, generic grids that collide by chance in millions of samples, not copies of eval tasks.
- **Answers are effectively absent.** Only one small eval answer appears, once in the official training set and once in NVARC synthetic.
- **Optional tightening:** for strictly clean training, drop samples containing any matched grid. That's <0.03% of the data, and `contamination_report.json` lists the affected tasks.
- **Unchanged:** NVARC's `nvarc_full` and `arc2_evaluation6` are still derived from the eval tasks by construction (see AGENTS.md). This check covers the subsets meant to be clean.
