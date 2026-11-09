#!/bin/bash
# E066: GPTQ-only export of a 2-bit candidate (E043 recipe: export_pruned.py --no-routed-experts, then
# make_lowbit_export.py), vision tower kept (--no-mtp drops only DSpark), every safetensors file <= 5 GB.
# The 64 GiB tool Engram is the row-chunked table of the HF tool E112 repo (split from the tool E128 export's table,
# the files the Kaggle E128 model ships; same Engram keep set, checked by mask hash), hardlinked into each export.
# Usage: export_candidate.sh engram                 download that Engram (needs $HF_TOKEN)
#        export_candidate.sh export <name> <keep.npy> <gptq dir> "<model name>" "<keep desc>" [--per-layer]
#        export_candidate.sh check  <name> <keep.npy> <gptq dir>   content check against the sources (no GPU)
# Output: /workspace/dsv4/export/<name>_gptq
set -euo pipefail
D=/workspace/dsv4
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
E=${ENGRAM_SRC:-$D/export/engram64_tool_chunked}  # ENGRAM_SRC: an export that already holds the same chunked table
SCHEMA='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
case $1 in
  engram)
    mkdir -p $E
    $D/tvenv/bin/python - $E <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download("chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit", local_dir=sys.argv[1], token=os.environ["HF_TOKEN"],
                  allow_patterns=["model-engram-*", "config.json", "e043_manifest.json", "model.safetensors.index.json"])
EOF
    $D/venv/bin/python - $E $D/data/e066/engram/e043tool_top64gib.npz <<'EOF'
import hashlib, json, sys
import numpy as np
e, mask = sys.argv[1:]
man = json.load(open(f"{e}/e043_manifest.json"))["engram"]
z = np.load(mask)
for hi, li in enumerate(sorted(man, key=int)):
    assert man[li]["mask_sha256"] == hashlib.sha256(z[f"layer{hi}"].tobytes()).hexdigest(), f"Engram layer {li} mask differs"
idx = json.load(open(f"{e}/model.safetensors.index.json"))["weight_map"]
import os
big = [f for f in set(idx.values()) if f.startswith("model-engram-") and os.path.getsize(f"{e}/{f}") > 5_000_000_000]
assert not big, f"Engram files above 5 GB: {big}"
print("ENGRAM_OK: the HF tool Engram uses the E066 Engram keep set, files <= 5 GB")
EOF
    ;;
  export)
    NAME=$2; K=$(realpath "$3"); G=$(realpath "$4"); MNAME=$5; DESC=$6; PL=${7:-}
    B=$D/export/${NAME}_base; X=$D/export/${NAME}_gptq
    rm -rf $B $X
    nice $D/venv/bin/python $S43/export_pruned.py --src $D/DeepSeek-V4.1-Flash-1017 --expert-keep $K --engram-from $E \
      --no-mtp --no-routed-experts $PL --shard-gb 4.9 --out $B
    nice $D/venv/bin/python $S43/make_lowbit_export.py --base $B --precomputed $G --precomputed-ids original --keep $K \
      --schema "$SCHEMA" --schema-w2 '{"group_size": 64}' --name "$MNAME" --keep-desc "$DESC" --out $X
    cp $G/*.json $X/gptq/
    rm -rf $B
    du -sh $X; echo "EXPORT_DONE $NAME" ;;
  check)
    NAME=$2; K=$(realpath "$3"); G=$(realpath "$4")
    nice $D/venv/bin/python $S43/check_lowbit_export.py --export $D/export/${NAME}_gptq --src $D/DeepSeek-V4.1-Flash-1017 \
      --gptq $G --keep $K
    echo "CHECK_OK $NAME" ;;
  *) echo "usage: export_candidate.sh engram | export <name> <keep> <gptq> <model name> <keep desc> [--per-layer] | check <name> <keep> <gptq>" >&2; exit 2 ;;
esac
