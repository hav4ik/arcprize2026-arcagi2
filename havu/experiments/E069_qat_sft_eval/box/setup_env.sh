#!/bin/bash
# E068 box setup: the E043 vLLM fork env from its lockfile + the E043 patches (observer/masks, compact Engram,
# Humming 2-bit), as E066 setup_env.sh (without the MoE-input dump patch, and without the tools venv).
set -euxo pipefail
D=/root/e068; cd $D
L=$D/e043/locks; P=$D/e043/patches
UV=$(command -v uv || echo /opt/instance-tools/bin/uv)
[ -x venv/bin/python ] || $UV venv --python /usr/bin/python3.12 venv
VIRTUAL_ENV=$D/venv $UV pip install -r $L/venv_vllm.freeze \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
SP=venv/lib/python3.12/site-packages/vllm
venv/bin/python $P/apply_e043_patches.py $SP
venv/bin/python $P/apply_compact_engram.py $SP
venv/bin/python $P/apply_e043_humming.py $SP
venv/bin/python -c "import importlib.metadata as m, torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), vllm.__version__, flashinfer.__version__, m.version('humming-kernels'))"
ls venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/
echo SETUP_DONE
