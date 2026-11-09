#!/bin/bash
# E062b (on L4 pod #2): install the same fork venv as pod #1 (/root/manh/install_manh.sh): the 4090-built wheel
# (b9ad670a67, sha256 cf111141) + freeze.txt + patches 01/02/03 over site-packages; checks the installed shas match
# pod #1's (model.py bbf92e10, chat_message.py def7ad87, types.py 39016a73, sparse_attn_indexer.py 3113fa17).
set -euo pipefail
cd /root/manh_setup
W=vllm-0.1.dev1+gb9ad670a6-cp312-cp312-linux_x86_64.whl
echo "cf111141781c24ee1e045100ab5ff44ccb2939f57b12366548585a75e1402973  $W" | sha256sum -c -
mkdir -p /root/manh
[ -x /root/manh/venv/bin/python ] || uv venv -q --python 3.12 /root/manh/venv
grep -v "^vllm @" freeze.txt > freeze_nov.txt
nice -n 10 uv pip install -q --python /root/manh/venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r freeze_nov.txt
nice -n 10 uv pip install -q --python /root/manh/venv/bin/python --no-deps $W
SP=/root/manh/venv/lib/python3.12/site-packages
for p in 01_engram_row_chunks.diff 02_reasoning_content_input.diff 03_indexer_deepgemm_guard_sm89.diff; do
  patch -p1 -d $SP --dry-run -s < $p && patch -p1 -d $SP -s < $p
done
sha256sum $SP/vllm/models/deepseek_v41/nvidia/model.py $SP/vllm/renderers/chat_utils/chat_message.py \
  $SP/vllm/renderers/chat_utils/types.py $SP/vllm/model_executor/layers/sparse_attn_indexer.py | cut -c1-12 | tr '\n' ' '; echo
/root/manh/venv/bin/python -c "import vllm, torch, importlib.metadata as m; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'flashinfer', m.version('flashinfer-python'), 'humming', m.version('humming-kernels'))"
echo MANH_INSTALL_OK
