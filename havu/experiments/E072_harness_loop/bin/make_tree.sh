#!/bin/bash
# /kaggle as the notebook sees it (adapted from phase_h make_kaggle_tree_owner.sh; E072). Usage: make_tree.sh <commit|rerun> <work dir>
# Env: KT_MODEL_OWNER KT_MODEL_SLUG KT_MODEL_DIR KT_SRC_DIR KT_WHEELS_DIR (required), KT_WORKLOAD_CH (challenges json for rerun)
set -euo pipefail
PH=$1; W=$2
: "${KT_MODEL_OWNER:?}" "${KT_MODEL_SLUG:?}" "${KT_MODEL_DIR:?}" "${KT_SRC_DIR:?}" "${KT_WHEELS_DIR:?}"
rm -rf /kaggle; mkdir -p /kaggle/input/datasets/nguyennguyen599 /kaggle/input/datasets/chankhavu /kaggle/input/competitions \
  /kaggle/input/models/$KT_MODEL_OWNER/$KT_MODEL_SLUG/pytorch/default
ln -s $KT_SRC_DIR /kaggle/input/datasets/nguyennguyen599/arc2-2026-src
ln -s /root/kin/cu13 /kaggle/input/datasets/nguyennguyen599/cu13-install
ln -s /root/kin/datasets/nguyennguyen599/unsloth-main/versions/15 /kaggle/input/datasets/nguyennguyen599/unsloth-main
ln -s /root/kin/datasets/chankhavu/dsv41-tools-kit/versions/5 /kaggle/input/datasets/chankhavu/dsv41-tools-kit
ln -s $KT_WHEELS_DIR /kaggle/input/datasets/nguyennguyen599/vllm-wheels
ln -s $KT_MODEL_DIR /kaggle/input/models/$KT_MODEL_OWNER/$KT_MODEL_SLUG/pytorch/default/1
C=/kaggle/input/competitions/arc-prize-2026-arc-agi-2; mkdir -p $C; cp /root/kin/comp/* $C/
if [ "$PH" = rerun ]; then cp "${KT_WORKLOAD_CH:?}" $C/arc-agi_test_challenges.json; fi
rm -rf "$W"; mkdir -p "$W"; ln -s "$W" /kaggle/working
printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo; chmod +x /usr/local/bin/sudo
ls -la /kaggle/input/datasets/*/ /kaggle/input/models/$KT_MODEL_OWNER/*/pytorch/default/ | grep -- "->" | awk '{print $9, $10, $11}'
echo "wheels: $(ls $KT_WHEELS_DIR/wheels/wheels | wc -l) ($(ls $KT_WHEELS_DIR/wheels/wheels | grep '^vllm-'))"; echo "competition: $(ls $C | tr '\n' ' ')"
echo "workload tasks: $(python3 -c "import json;print(len(json.load(open('$C/arc-agi_test_challenges.json'))))")"
