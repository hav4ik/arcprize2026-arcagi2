# Code embedded in dsv41-tools-submit-mtf Version 6

The exact files packed into the notebook's cell 0b (`_MT`: base64 of a .tar.gz), checked there by sha256:

| file | sha256 | installed as |
|---|---|---|
| `solver_dsv41.py` | 4b344011... | `$KIT/solver_dsv41.py` |
| `run_deepseek_tools.py` | a8a5b665... | `$KIT/repo/evals/run_deepseek_tools.py` |

`evals/run_deepseek_tools.py` in the repo has moved on since (E047 options); rebuild the notebook from these copies:
`kaggle/dsv41_tools/build_frozen_mt.py --slug dsv41-tools-submit-mtf --workers 36 --budget 0.5 --gate 0.6
--solver kaggle/dsv41_tools/pinned_mtf_v6/solver_dsv41.py --harness kaggle/dsv41_tools/pinned_mtf_v6/run_deepseek_tools.py`
