# E015: Hodel's arc-dsl and ARC-AGI-2 (no API cost)

## Question
How useful is Michael Hodel's arc-dsl (https://github.com/michaelhodel/arc-dsl) for ARC-AGI-2? It could serve as a helper library, as a search space, for data generation, or for token compression.

## Third-party code
- **Clone:** `vendor/arc-dsl` is pinned at **`635de4902a5fb4e376f27333feaa396d3f5dfdcb`** (2024-10-11, "Merge pull request #13 ... fix-solve_6a1e5592"). It is listed in `.gitignore`; do not copy it elsewhere.
- **Read before running:** `dsl.py` and `solvers.py` are pure functions. They import only `arc_types`, `constants` and `dsl`, and do no I/O.
- **How it is run:** only through our own scripts in this directory, with `.venv/bin/python -I`, which add the vendor path explicitly (`common.py`).

## Results
See `empirical.md` (full numbers) and `survey.md` (literature survey, if present).

## Reproduce (CPU only, about 45 CPU-min in total)
```bash
P=".venv/bin/python -I experiments/E015_dsl_analysis"
$P/inventory.py; $P/carryover.py; $P/extract_programs.py; $P/summarise_review.py; $P/dsl_solvers.py
$P/search.py --set arc2_eval --budget 10 --workers 20; $P/search.py --set arc1_eval --budget 3 --workers 20
$P/solver_transfer.py
```
ARC-AGI-1 data is read from `/home/vu/Workspace/auto-neurogolf/ARC-AGI/data`, a clone of fchollet/ARC-AGI @ 399030444e0a.
