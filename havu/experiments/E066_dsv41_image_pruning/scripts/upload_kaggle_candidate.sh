#!/bin/bash
# E066: upload a candidate export as a PRIVATE Kaggle Model (framework PyTorch, instance "default"), laid out like the
# E128 Kaggle Model: plain files; subfolders (gptq/, serving/, encoding/, inference/) go up as tar archives that
# Kaggle unpacks. Then compare Kaggle's file list with the staged folder by name and size (E043 kaggle_verify_model.py).
# Token: $KAGGLE_API_TOKEN in the environment only (never on the command line).
# Usage: upload_kaggle_candidate.sh <export name> <owner> <slug> "<title>" "<subtitle>"
set -uo pipefail
[ -n "${KAGGLE_API_TOKEN:-}" ] || { echo "KAGGLE_API_TOKEN not set"; exit 1; }
D=/workspace/dsv4
NAME=$1; OWNER=$2; SLUG=$3; TITLE=$4; SUB=$5
X=$D/export/${NAME}_gptq; S=$D/export/kaggle_${NAME}; M=$D/export/kaggle_${NAME}_meta
K=$D/tvenv/bin/kaggle
export TMPDIR=$D/export/kgtmp; mkdir -p $TMPDIR $M
rm -rf $S; mkdir -p $S
for f in $X/*; do b=$(basename $f); [ -d $f ] && cp -al $f $S/$b || ln $f $S/$b; done
find $S -name __pycache__ -prune -exec rm -rf {} +; rm -rf $S/.cache
$D/tvenv/bin/python - $S $M "$OWNER" "$SLUG" "$TITLE" "$SUB" <<'PY'
import json, sys
s, m, owner, slug, title, sub = sys.argv[1:]
card = open(f"{s}/README.md").read()
json.dump({"ownerSlug": owner, "title": title, "slug": slug, "subtitle": sub, "isPrivate": True, "description": card,
           "publishTime": "", "provenanceSources": ""}, open(f"{m}/model-metadata.json", "w"), indent=1)
json.dump({"ownerSlug": owner, "modelSlug": slug, "instanceSlug": "default", "framework": "pyTorch",
           "overview": sub, "usage": card, "licenseName": "MIT", "fineTunable": False, "trainingData": [],
           "modelInstanceType": "Unspecified", "baseModelInstanceId": 0, "externalBaseModelUrl": ""},
          open(f"{s}/model-instance-metadata.json", "w"), indent=1)
print("metadata written")
PY
if $K models get $OWNER/$SLUG -p $M/existing > /dev/null 2>&1; then echo "model exists"; else
  $K models create -p $M 2>&1 | grep -v Warning || exit 1; fi
date -u; echo "uploading $S -> $OWNER/$SLUG/pyTorch/default"
$K models instances create -p $S -r tar 2>&1 | grep -v -E "Warning|it/s\]|B/s\]" | grep -v "^\s*$"
rc=${PIPESTATUS[0]}
rm -rf $TMPDIR/*
date -u; echo "instance create rc=$rc"
for i in $(seq 1 40); do
  $D/tvenv/bin/python $D/repo/experiments/E043_dsv41_real_pruning/scripts/kaggle_verify_model.py $OWNER/$SLUG/pyTorch/default/1 $S \
    && { echo "KAGGLE_UPLOAD_VERIFIED $OWNER/$SLUG"; exit 0; }
  sleep 60
done
echo KAGGLE_UPLOAD_NOT_VERIFIED; exit 1
