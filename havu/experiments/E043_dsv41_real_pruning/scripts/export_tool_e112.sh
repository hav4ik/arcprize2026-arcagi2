#!/bin/bash
# E043: GPTQ-only export of tool-use E112 2-bit, every safetensors file <= 5 GB (user rule, 2026-10-24).
# Built in /dev/shm (/workspace has ~60 GB free): (1) the 64 GiB tool Engram of the tool E128 export, re-sharded into
# <= 5 GB row chunks (read through symlinks, no copy); (2) export_pruned.py --engram-from that dir (hardlinks) with
# 4.9 GB backbone shards; (3) make_lowbit_export.py with the E112 GPTQ tensors; (4) size check.
set -euo pipefail
cd /workspace/dsv4
K=data/reap/keep_tool/keep_contrib_dec_sp0_w10_e112.npy
O=/dev/shm/e043/gptq_tool_e112_u2
SRC=export/tool_e128_w10_eng64_gptq
E=/dev/shm/e043/export/engram64_tool_chunked
B=/dev/shm/e043/export/tool_e112_w10_eng64_base
X=/dev/shm/e043/export/tool_e112_w10_eng64_gptq
until [ "$(ls $O/*.json 2>/dev/null | wc -l)" = 40 ] && ! pgrep -f "gptq_experts.py --calx /dev/shm/e043/calx_tool_e112" > /dev/null; do sleep 30; done
date -u; echo "GPTQ done; chunking the tool Engram"
if [ ! -f $E/model.safetensors.index.json ]; then
  rm -rf $E; mkdir -p $E
  for f in $SRC/model-engram-*.safetensors; do ln -s $(readlink -f $f) $E/$(basename $f); done
  cp $SRC/config.json $SRC/e043_manifest.json $E/
  tvenv/bin/python - $SRC $E <<'EOF'
import json, sys
src, dst = sys.argv[1:]
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
eng = {k: v for k, v in wm.items() if v.startswith("model-engram-")}
json.dump({"metadata": {"total_size": 0}, "weight_map": eng}, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
print(len(eng), "engram tensors")
EOF
  tvenv/bin/python code/scripts/reshard_safetensors.py $E
fi
date -u; echo "export_pruned (backbone + router slices, Engram hardlinked)"
rm -rf $B $X
venv/bin/python code/scripts/export_pruned.py --src /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017 --expert-keep $K \
  --engram-from $E --text-only --no-routed-experts --shard-gb 4.9 --out $B
date -u; echo "make_lowbit_export"
venv/bin/python code/scripts/make_lowbit_export.py --base $B --precomputed $O --precomputed-ids original --keep $K \
  --schema '{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}' \
  --schema-w2 '{"group_size": 64}' --name "E112 2-bit GPTQ, tool use" \
  --keep-desc "tool-use keep set: contribution score on tool rollouts, tool-call-block weight 10" --out $X
cp $O/*.json $X/gptq/
rm -rf $B
python3 - $X <<'EOF'
import json, os, sys
d = sys.argv[1]
sizes = {os.path.relpath(os.path.join(r, f), d): os.path.getsize(os.path.join(r, f))
         for r, _, fs in os.walk(d) for f in fs if f.endswith(".safetensors")}
big = {k: v for k, v in sizes.items() if v > 5_000_000_000}
idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
missing = sorted(set(idx.values()) - set(sizes))
print(f"{len(sizes)} safetensors, largest {max(sizes.values()) / 1e9:.3f} GB, total {sum(sizes.values()) / 1e9:.2f} GB")
assert not big, f"files above 5 GB: {big}"
assert not missing, f"index files missing: {missing}"
print("SIZE-CHECK-OK")
EOF
du -sh $X; df -h /dev/shm | tail -1
echo EXPORT-TOOL-E112-DONE
