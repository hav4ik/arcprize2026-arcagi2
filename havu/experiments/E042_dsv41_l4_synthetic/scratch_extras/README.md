# Scratchpad extras (2026-10-24..26)

- `pr154.diff`: full diff of fork PR 154 (the kernel part is in `../kaggle/kit/pr154_kernel.diff`).
- `fork_ced/`: the CED (bounded replay) experiment files patched into the fork; rejected as approximate (E042 README).
- `run_kvfp4.sh`: the 10-24 FP4 KV test (`nvfp4_ds_mla`, refused by the fork; `fp8` already selects mixed FP8/FP4 KV).
- `budget12h.py`, `gpu_windows.py`, `metrics_windows.py`, `kvwin*.py`: analysis helpers used for the Kaggle deadline and KV-window studies.
