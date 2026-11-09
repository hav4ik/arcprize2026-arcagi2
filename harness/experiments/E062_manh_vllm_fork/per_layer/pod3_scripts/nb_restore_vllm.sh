#!/bin/bash
# after the DCP4 server is stopped: vLLM files back to the wheel's originals (cell 2r then patches them as on Kaggle)
set -euo pipefail
# vLLM files back to the wheel's originals, so cell 2r patches them as on Kaggle
SP=$(/root/b32m/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cd $SP
patch -R -p1 --no-backup-if-mismatch < /root/e062pl/dynamic_count_model_f506.diff
patch -R -p1 --no-backup-if-mismatch < /root/e062pl/dynamic_count_dsv4_topk.diff
sha256sum vllm/models/deepseek_v41/nvidia/model.py vllm/model_executor/layers/fused_moe/router/dsv4_topk.py | cut -c1-8,65-
echo "RESTORE_DONE $(date -u +%T)"
