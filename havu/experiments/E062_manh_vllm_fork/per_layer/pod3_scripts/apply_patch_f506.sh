#!/bin/bash
# dynamic-count-moe runtime patch on Manh's wheel gf506afa210 (vllm-wheels v10): model.py 029aa62e -> ad4ac0fd
# (b32m_patches/dynamic_count_model_f506.diff), dsv4_topk.py 7e1e101a -> ab32ea7e; base hashes checked first.
set -euo pipefail
V=/root/b32m/venv; SP=$($V/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
R=vllm/model_executor/layers/fused_moe/router/dsv4_topk.py; M=vllm/models/deepseek_v41/nvidia/model.py
h() { sha256sum "$SP/$1" | cut -c1-8; }
echo "before $(h $R) $R"; echo "before $(h $M) $M"
[ "$(h $R)" = 7e1e101a ] || { echo "UNEXPECTED BASE dsv4_topk $(h $R)"; exit 3; }
[ "$(h $M)" = 029aa62e ] || { echo "UNEXPECTED BASE model.py $(h $M)"; exit 3; }
cd $SP
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/dynamic_count_dsv4_topk.diff
patch -p1 --forward --no-backup-if-mismatch < /root/e062pl/dynamic_count_model_f506.diff
echo "after  $(h $R) $R"; echo "after  $(h $M) $M"
[ "$(h $R)" = ab32ea7e ] && [ "$(h $M)" = ad4ac0fd ] || { echo "PATCHED HASH MISMATCH"; exit 4; }
find $SP/vllm/model_executor/layers/fused_moe/router $SP/vllm/models/deepseek_v41/nvidia -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
$V/bin/python -c "import vllm.models.deepseek_v41.nvidia.model as m; print('IMPORT_OK', m.routed_experts_per_layer)"
