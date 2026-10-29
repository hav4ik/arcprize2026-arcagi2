#!/bin/bash
# Map /kaggle as the notebook sees it on Kaggle (read paths symlinked to the verified pod copies; /kaggle/working fresh).
# Usage: make_kaggle_tree.sh <phase: commit|rerun> <work dir>
set -euo pipefail
PH=$1; W=$2
rm -rf /kaggle; mkdir -p /kaggle/input/datasets/nguyennguyen599 /kaggle/input/datasets/chankhavu /kaggle/input/competitions \
  /kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-img-e72-152-avg120-2bit/pytorch/default
ln -s /root/manh_ds/.khcache/datasets/nguyennguyen599/arc2-2026-src/versions/22 /kaggle/input/datasets/nguyennguyen599/arc2-2026-src
ln -s /root/kin/cu13 /kaggle/input/datasets/nguyennguyen599/cu13-install
ln -s /root/kin/datasets/nguyennguyen599/unsloth-main/versions/15 /kaggle/input/datasets/nguyennguyen599/unsloth-main
ln -s /root/kin/datasets/chankhavu/dsv41-tools-kit/versions/5 /kaggle/input/datasets/chankhavu/dsv41-tools-kit
# vllm-wheels v10 = v8's 294 top-level wheels (only the vllm wheel differs, per the API listing) + v10's vllm wheel
VW=/root/kin/vllm_wheels_v10/wheels/wheels; if [ ! -d $VW ]; then mkdir -p $VW; for f in /root/b32m/dl/wheels/wheels/*.whl; do
  case "$(basename $f)" in vllm-*) ;; *) ln "$f" "$VW/" ;; esac; done; ln /root/b32m/dl_v10/wheels/wheels/vllm-*.whl "$VW/"; fi
ln -s /root/kin/vllm_wheels_v10 /kaggle/input/datasets/nguyennguyen599/vllm-wheels
ln -s /root/models/e72c152 /kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-img-e72-152-avg120-2bit/pytorch/default/1
C=/kaggle/input/competitions/arc-prize-2026-arc-agi-2; mkdir -p $C; cp /root/kin/comp/* $C/
if [ "$PH" = rerun ]; then cp /root/data/evalx2t_s2026_challenges.json $C/arc-agi_test_challenges.json; fi
rm -rf "$W"; mkdir -p "$W"; ln -s "$W" /kaggle/working
printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo; chmod +x /usr/local/bin/sudo
ls -la /kaggle/input/datasets/*/ /kaggle/input/models/chankhavu/*/pytorch/default/ | grep -- "->" | awk '{print $9, $10, $11}'
echo "wheels: $(ls $VW | wc -l) ($(ls $VW | grep '^vllm-'))"; echo "competition: $(ls $C | tr '\n' ' ')"
