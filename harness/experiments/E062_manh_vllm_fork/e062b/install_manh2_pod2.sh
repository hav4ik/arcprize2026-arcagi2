#!/bin/bash
# E062b (on L4 pod #2): a second fork venv /root/manh2/venv for arm B: the SM89 wheel of the local merge
# arc/e128-kaggle-main5e1b (95376a1, built on the 48 GB box) on the same pinned deps as /root/manh/venv (freeze.txt).
# FREEZE= the pinned deps file (pod #2: /root/manh_setup/freeze_nov.txt, pod #1: /root/manh/freeze_nov.txt).
# No patches: the merge contains 01/02/03 and the autotune (off by default).
set -euo pipefail
cd /root/manh2_setup
W=$(ls vllm-*.whl | head -1); sha256sum -c wheel.sha256
[ -x /root/manh2/venv/bin/python ] || { mkdir -p /root/manh2; uv venv -q --python 3.12 /root/manh2/venv; }
nice -n 10 uv pip install -q --python /root/manh2/venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r ${FREEZE:-/root/manh_setup/freeze_nov.txt}
nice -n 10 uv pip install -q --python /root/manh2/venv/bin/python --no-deps $W
/root/manh2/venv/bin/python -c "import vllm, torch, vllm.envs as e, importlib.metadata as m; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'humming', m.version('humming-kernels'), 'row_shard env', 'VLLM_DSA_DECODE_INDEXER_ROW_SHARD' in e.environment_variables, 'autotune env', 'VLLM_COMM_AUTOTUNE' in e.environment_variables)"
echo MANH2_INSTALL_OK
