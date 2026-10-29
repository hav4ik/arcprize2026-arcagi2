#!/bin/bash
# Pod 2: switch Manh's venv to vllm-wheels v10 (gf506afa210) - vllm only, offline, no deps - then the E066 runtime
# patches for that base (kaggle/dsv41_tools/b32m_patches: dynamic_count_model_f506.diff 029aa62e -> ad4ac0fd,
# dynamic_count_dsv4_topk.diff 7e1e101a -> ab32ea7e), hash-checked before and after.
set -euo pipefail
V=/root/b32m/venv; W=/root/b32m/dl_v10/wheels/wheels/vllm-0.32.1rc1.dev0+gf506afa210-cp313-cp313-linux_x86_64.whl
echo "wheel sha $(sha256sum $W | cut -c1-16)"
UV_OFFLINE=1 uv pip install -q --python $V/bin/python --no-deps --reinstall "$W"
$V/bin/python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), 'torch', m.version('torch'), 'triton', m.version('triton'), 'humming', m.version('humming-kernels'))"
SP=$($V/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
M=vllm/models/deepseek_v41/nvidia/model.py; R=vllm/model_executor/layers/fused_moe/router/dsv4_topk.py
cd $SP
echo "base model.py $(sha256sum $M | cut -c1-16) (expect 029aa62e383332aa)  dsv4_topk $(sha256sum $R | cut -c1-16) (expect 7e1e101ab2fb266b)"
[ "$(sha256sum $M | cut -c1-16)" = 029aa62e383332aa ] && [ "$(sha256sum $R | cut -c1-16)" = 7e1e101ab2fb266b ]
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/dynamic_count_model_f506.diff
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/dynamic_count_dsv4_topk.diff
echo "patched model.py $(sha256sum $M | cut -c1-16) (expect ad4ac0fd9bc604fe)  dsv4_topk $(sha256sum $R | cut -c1-16) (expect ab32ea7e2f3a6976)"
[ "$(sha256sum $M | cut -c1-16)" = ad4ac0fd9bc604fe ] && [ "$(sha256sum $R | cut -c1-16)" = ab32ea7e2f3a6976 ]
find $SP/vllm -name __pycache__ -path "*deepseek_v41*" -prune -exec rm -rf {} + ; find $SP/vllm/model_executor/layers/fused_moe/router -name __pycache__ -prune -exec rm -rf {} +
$V/bin/python -c "import vllm.models.deepseek_v41.nvidia.model as m; print('IMPORT_OK', m.max_routed_experts)" 2>&1 | grep IMPORT
echo SWITCH_V10_OK
