#!/bin/bash
# vllm-wheels v9 (current at B32-M v3's submission, 2026-10-28 23:37 UTC; its listing equals v8's) at /root/b32m/dl9:
# v10's 293 unchanged wheels as hardlinks plus v9's vllm-0.32.1rc1.dev0+g5e1bccd07c wheel and .gitattributes, pinned
# to versions/9 (token in the environment or on stdin). The 4 nested source dirs are left out, as for v10/v11.
set -euo pipefail
[ -n "${KAGGLE_API_TOKEN:-}" ] || read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
D=/root/b32m/dl9; mkdir -p $D/wheels/wheels
for f in /root/b32m/dl/wheels/wheels/*.whl; do case "$f" in *vllm-0.32.1rc1.dev0+gf506afa210*) ;; *) ln -f "$f" $D/wheels/wheels/ ;; esac; done
cd /root/b32m && /root/kgl/bin/python -I dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/9 /root/e066/b32mv3/v9_min.tsv $D > /root/e066/b32mv3/dl_v9.log 2>&1
unset KAGGLE_API_TOKEN; rm -rf $D/.khcache
tail -1 /root/e066/b32mv3/dl_v9.log; ls $D/wheels/wheels/*.whl | wc -l; ls $D/wheels/wheels/ | grep vllm-0; sha256sum $D/wheels/wheels/vllm-0.32.1rc1.dev0+g5e1bccd07c-*.whl | cut -c1-16
