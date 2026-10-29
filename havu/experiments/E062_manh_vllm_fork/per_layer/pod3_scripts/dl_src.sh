#!/bin/bash
# Download one pinned version of nguyennguyen599/arc2-2026-src (kagglehub cache /root/khsrc), token read from stdin.
# Usage: printf '%s\n' "$TOKEN" | dl_src.sh <version>
set -euo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
for i in 1 2 3; do
  KAGGLEHUB_CACHE=/root/khsrc /root/kgl/bin/python -I -c "import kagglehub; print('SRC', kagglehub.dataset_download('nguyennguyen599/arc2-2026-src/versions/$1'))" > /root/e066/r3/dl_src_v$1.log 2>&1 && break
  sleep $((i * 20))
done
unset KAGGLE_API_TOKEN
grep SRC /root/e066/r3/dl_src_v$1.log; D=/root/khsrc/datasets/nguyennguyen599/arc2-2026-src/versions/$1
ls $D | head; grep -n "supervise" $D/dsv41_vllm.py | head -5; sha256sum $D/dsv41_vllm.py | cut -c1-16
