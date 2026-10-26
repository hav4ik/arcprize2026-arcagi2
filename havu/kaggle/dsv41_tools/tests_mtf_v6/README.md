# Tests written by the 2026-10-26 deep audit of dsv41-tools-submit-mtf Version 6

44 tests (all passed against `../pinned_mtf_v6/`): `test_harness_new.py` (multi-grid submit, shadowed builtins,
`_plain`, context limit / ContextFull fallbacks, `--multi-budget` scaling, budget notes, train-check labels),
`test_solver_new.py` (env parsing, `split_multi`, `pick`, `kv_admits`, `main()` with a fake server and attempts),
`test_notebook_cells.py` (settings cell + cell 0b in commit / rerun, OK / wrong setting / broken kit / fallback).

`common.py` expects the audit's layout next to this folder: `envB/evals/` (the repo's `evals/` with the pinned
harness a8a5b665 and the kit's train_check 44c4d210) and `v3/` (the extracted Kaggle `dsv41-tools-kit`); it asserts
those hashes. Recreate that layout to run them as is. Porting them against the live harness and solver was item T4 of `../next_candidate_fixes.md`, done on 2026-10-26:
`evals/tests/test_dsv41_harness.py` and `../tests/` (solver, notebook cells, tooling), with the fixes' regression tests.

`e2e/`: the end-to-end dry-run tools (fake token-exact DSML server `fake_server.py`, notebook runner `run_nb.py`, scenarios `run_scn.sh`, `validate.py`, heavy-set builder `make_heavy.py`, `close_test.py`, `metrics_parse_test.py`). They use scratch paths from the audit; adjust when porting.
