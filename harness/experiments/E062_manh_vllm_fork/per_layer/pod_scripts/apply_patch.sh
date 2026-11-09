#!/bin/bash
# dynamic-count-moe runtime patch onto the installed wheel (5e1bccd07c): router whitelist + per-layer model hunk.
set -euo pipefail
V=/root/b32m/venv; SP=$($V/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
R=vllm/model_executor/layers/fused_moe/router/dsv4_topk.py; M=vllm/models/deepseek_v41/nvidia/model.py
echo "site-packages: $SP"
for f in $R $M; do echo "before $(sha256sum $SP/$f | cut -c1-16) $f"; done
cd $SP
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/router.diff
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/model.diff
for f in $R $M; do echo "after  $(sha256sum $SP/$f | cut -c1-16) $f"; done
find $SP/vllm -name "__pycache__" -path "*fused_moe/router*" -prune -exec rm -rf {} + 2>/dev/null || true
find $SP/vllm/models/deepseek_v41/nvidia -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
$V/bin/python -c "import vllm.models.deepseek_v41.nvidia.model as m; print('IMPORT_OK', m.routed_experts_per_layer, m.max_routed_experts)"
