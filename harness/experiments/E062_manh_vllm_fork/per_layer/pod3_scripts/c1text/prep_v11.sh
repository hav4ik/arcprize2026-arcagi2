#!/bin/bash
# vllm-wheels v11 tree at /root/b32m/dl11: v10's 293 unchanged wheels as hardlinks (v11 = v10 by name and size except
# the vllm wheel), plus v11's vllm wheel (g8cf0c0664b) and .gitattributes downloaded pinned to versions/11 (token on stdin).
set -euo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
D=/root/b32m/dl11; mkdir -p $D/wheels/wheels
for f in /root/b32m/dl/wheels/wheels/*.whl; do case "$f" in *vllm-0.32.1rc1.dev0+gf506afa210*) ;; *) ln -f "$f" $D/wheels/wheels/ ;; esac; done
cd /root/b32m && /root/kgl/bin/python -I dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/11 /root/e066/c1text/v11_min.tsv $D > /root/e066/c1text/dl_v11.log 2>&1
unset KAGGLE_API_TOKEN
tail -3 /root/e066/c1text/dl_v11.log; ls $D/wheels/wheels/*.whl | wc -l; ls $D/wheels/wheels/ | grep vllm-0; sha256sum $D/wheels/wheels/vllm-0.32.1rc1.dev0+g8cf0c0664b-*.whl | cut -c1-12; rm -rf $D/.khcache
