#!/bin/bash
# E066 box setup (run in /workspace/dsv4 with repo/ synced): the E043 vLLM fork env from its lockfile, the E043
# patches in order (observer + masks, compact Engram, Humming 2-bit, MoE-input dump), and the tools env.
# Used on Vast 58209054 (pruning box) on 2026-10-29: ~10 min.
set -euxo pipefail
cd /workspace/dsv4
L=repo/experiments/E043_dsv41_real_pruning/locks
P=repo/experiments/E043_dsv41_real_pruning/patches
uv venv --python 3.12 venv
VIRTUAL_ENV=/workspace/dsv4/venv uv pip install -r $L/venv_vllm.freeze \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
SP=venv/lib/python3.12/site-packages/vllm
venv/bin/python $P/apply_e043_patches.py $SP
venv/bin/python $P/apply_compact_engram.py $SP
venv/bin/python $P/apply_e043_humming.py $SP
venv/bin/python $P/apply_e043_dumpx.py $SP
uv venv --python 3.12 tvenv
VIRTUAL_ENV=/workspace/dsv4/tvenv uv pip install -r $L/tvenv.freeze
VIRTUAL_ENV=/workspace/dsv4/tvenv uv pip install requests pyarrow numpy
echo SETUP_DONE
