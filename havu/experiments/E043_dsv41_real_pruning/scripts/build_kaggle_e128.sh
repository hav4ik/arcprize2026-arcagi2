#!/bin/bash
# E043: stage the tool E128 2-bit export for a private Kaggle Model with every safetensors file <= 5 GB.
# - Engram: hardlinks to the E112 export's 16 row-chunk files (re-chunked from this E128 export's own Engram files and
#   sha256-checked then; the check below repeats it against the originals);
# - backbone: the 2 shards resharded to <= 5 GB (one was 5.03 GB);
# - gptq/: a symlink to the export's directory (the kaggle CLI tars it during the upload; Kaggle unpacks it);
# - configs, tokenizer, encoding/, inference/, LICENSE copied; serving/ = the patches this layout needs; README = card.
# Then verify_reshard_equiv.py against the original export.
set -euo pipefail
cd /workspace/dsv4
SRC=export/tool_e128_w10_eng64_gptq
CH=/dev/shm/e043/export/tool_e112_w10_eng64_gptq
S=/dev/shm/e043/kaggle_e128
B=/dev/shm/e043/kaggle_e128_backbone
rm -rf $S $B; mkdir -p $S $B
# backbone: reshard symlinks to the two shards (index restricted to them)
for f in $SRC/model-0000?.safetensors; do ln -s $(readlink -f $f) $B/$(basename $f); done
tvenv/bin/python - $SRC $B <<'EOF'
import json, sys
src, dst = sys.argv[1:]
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
bb = {k: v for k, v in wm.items() if not v.startswith("model-engram-")}
json.dump({"metadata": {"total_size": 0}, "weight_map": bb}, open(f"{dst}/model.safetensors.index.json", "w"))
print(len(bb), "backbone tensors in", sorted(set(bb.values())))
EOF
tvenv/bin/python code/scripts/reshard_safetensors.py $B
# assemble
for f in $B/model-*-of-*.safetensors; do mv $f $S/; done
for f in $CH/model-engram-*-of-00016.safetensors; do ln $f $S/$(basename $f); done
ln -s $(readlink -f $SRC/gptq) $S/gptq
for f in config.json e043_manifest.json tokenizer.json tokenizer_config.json LICENSE; do cp $SRC/$f $S/; done
cp -r $SRC/encoding $SRC/inference $S/
mkdir -p $S/serving
cp code/patches/apply_compact_engram.py code/patches/apply_e043_humming.py code/patches/e043_humming.py $S/serving/
cp code/scripts/README_tool_e128_kaggle.md $S/README.md
tvenv/bin/python - $B $CH $S <<'EOF'
import json, sys
bdir, chdir, sdir = sys.argv[1:]
bb = json.load(open(f"{bdir}/model.safetensors.index.json"))
ch = json.load(open(f"{chdir}/model.safetensors.index.json"))["weight_map"]
wm = dict(bb["weight_map"])
wm.update({k: v for k, v in ch.items() if v.startswith("model-engram-")})
meta = {"e043_reshard": {"max_bytes": 5000000000, "row_chunk_suffix": ".rows_<start>"}}
json.dump({"metadata": meta, "weight_map": dict(sorted(wm.items()))}, open(f"{sdir}/model.safetensors.index.json", "w"), indent=2)
print(len(wm), "tensors in the merged index")
EOF
rm -rf $B
ls -la $S | head -40
tvenv/bin/python code/scripts/verify_reshard_equiv.py $SRC $S --allow-diff README.md
du -sbL $S | cut -f1
echo BUILD-KAGGLE-E128-DONE
