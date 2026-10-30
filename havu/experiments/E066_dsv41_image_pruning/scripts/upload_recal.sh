#!/bin/bash
# E066: upload the fixed-path re-prune (recal_chain.sh, export/c1f_gptq) to a private HF model repo WITHOUT the
# 64 GiB Engram table (user 2026-10-30: "without engram tables is fine"). Stages hardlinks of every export file except
# model-engram-*, the card (cards/README_c1f.md: {SIZE}, {SHIFT} filled) and serving/ (the E043 patches), uploads with
# retries and compares the repo with the staged folder by name and size. Token: read from stdin (never on disk or the
# command line).   Usage: upload_recal.sh <repo> < token
set -uo pipefail
read -r HF_TOKEN; export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "no token on stdin"; exit 1; }
D=/workspace/dsv4
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
E43=$D/repo/experiments/E043_dsv41_real_pruning
REPO=$1; X=$D/export/c1f_gptq; H=$D/export/c1f_hf
grep -q CHECK_OK $D/logs/recal_check.log || { echo "export check not passed"; exit 1; }
rm -rf $H; mkdir -p $H
(cd $X && find . -type f ! -name 'model-engram-*' ! -path './.cache/*' -printf '%P\n') | while read -r f; do
  mkdir -p "$H/$(dirname "$f")"; ln "$X/$f" "$H/$f"
done
mkdir -p $H/serving
cp $E43/patches/apply_compact_engram.py $E43/patches/apply_e043_humming.py $E43/patches/e043_humming.py $H/serving/
rm -f $H/README.md
$D/tvenv/bin/python - $S/cards/README_c1f.md $H/README.md $D/data/e066/keep_shift.json $(du -sb $H | cut -f1) <<'PY'
import json, sys
card, out, ks, size = sys.argv[1:]
o = json.load(open(ks))["overlap_new_vs_c1"]
s = open(card).read().replace("{SIZE}", f"{int(size):,}").replace(
    "{SHIFT}", f"Versus candidate 1's keep set: {o['swapped_mean']} of 128 experts per layer swapped on average (max {128 - o['min']}).")
assert "{SIZE}" not in s and "{SHIFT}" not in s
open(out, "w").write(s)
PY
[ $? = 0 ] || { echo CARD_FAILED; exit 1; }
date -u; echo "uploading $H -> $REPO ($(du -sh $H | cut -f1))"
for i in 1 2 3; do
  $D/tvenv/bin/python $E43/scripts/hf_upload.py --repo $REPO --type model --path $H \
    --message "E066 re-prune with causal image attention: 2-bit GPTQ, vision tower kept, no Engram table" && break
  echo "upload attempt $i failed; retrying in 60 s"; sleep 60
done
date -u
if $D/tvenv/bin/python $E43/scripts/hf_verify_upload.py --repo $REPO --path $H; then echo "UPLOAD_DONE $REPO $(du -sb $H | cut -f1)"
else echo "UPLOAD_FAILED $REPO"; exit 1; fi
