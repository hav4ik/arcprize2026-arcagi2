#!/bin/bash
# E043: upload the staged tool E128 2-bit checkpoint (build_kaggle_e128.sh) as a private Kaggle Model
# chankhavu/dsv41-flash-1017-arc-tool-e128-2bit (framework pyTorch, instance "default"), then compare Kaggle's file list
# with the staged folder by name and size. Subfolders (gptq/, serving/, encoding/, inference/) go up as tar archives
# that Kaggle unpacks (checked on a test model); the archive is built under $TMPDIR in RAM.
# Token: $KAGGLE_API_TOKEN, else the first line of stdin (never on the command line).
set -uo pipefail
[ -n "${KAGGLE_API_TOKEN:-}" ] || read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
cd /workspace/dsv4
S=/dev/shm/e043/kaggle_e128
M=/dev/shm/e043/kaggle_e128_meta
SLUG=dsv41-flash-1017-arc-tool-e128-2bit
K=tvenv/bin/kaggle
export TMPDIR=/dev/shm/e043/kgtmp; mkdir -p $TMPDIR $M
grep -q BUILD-KAGGLE-E128-DONE logs/build_kaggle_e128.log || { echo "staging not built/verified"; exit 1; }
tvenv/bin/python - $S $M $SLUG <<'EOF'
import json, sys
s, m, slug = sys.argv[1:]
card = open(f"{s}/README.md").read()
json.dump({"ownerSlug": "chankhavu", "title": "DeepSeek-V4.1-Flash-1017 ARC tool E128 2-bit", "slug": slug,
           "subtitle": "E043: 128 of 384 experts, 2-bit GPTQ, compact Engram; tool-integrated ARC-AGI-2",
           "isPrivate": True, "description": card, "publishTime": "", "provenanceSources": ""},
          open(f"{m}/model-metadata.json", "w"), indent=1)
json.dump({"ownerSlug": "chankhavu", "modelSlug": slug, "instanceSlug": "default", "framework": "pyTorch",
           "overview": "DeepSeek-V4.1-Flash-1017 pruned to 128 routed experts per layer with 2-bit GPTQ experts "
                       "(Humming layout) and a compact 64 GiB Engram; every safetensors file <= 5 GB.",
           "usage": card, "licenseName": "MIT", "fineTunable": False, "trainingData": [],
           "modelInstanceType": "Unspecified", "baseModelInstanceId": 0, "externalBaseModelUrl": ""},
          open(f"{s}/model-instance-metadata.json", "w"), indent=1)
print("metadata written")
EOF
if $K models get chankhavu/$SLUG -p $M/existing > /dev/null 2>&1; then echo "model exists"; else
  $K models create -p $M 2>&1 | grep -v Warning || exit 1; fi
date -u; echo "uploading $S (TMPDIR=$TMPDIR)"
$K models instances create -p $S -r tar 2>&1 | grep -v "Warning\|it/s\]\|B/s\]" | grep -v "^\s*$"
rc=${PIPESTATUS[0]}
rm -rf $TMPDIR/*
date -u; echo "instance create rc=$rc"
for i in $(seq 1 30); do
  tvenv/bin/python code/scripts/kaggle_verify_model.py chankhavu/$SLUG/pyTorch/default/1 $S && { echo UPLOAD-KAGGLE-E128-DONE; exit 0; }
  sleep 60
done
echo UPLOAD-KAGGLE-E128-NOT-VERIFIED; exit 1
