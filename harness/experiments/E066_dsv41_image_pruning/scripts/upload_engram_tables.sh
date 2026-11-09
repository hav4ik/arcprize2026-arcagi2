#!/bin/bash
# E066: stage the two MXFP4 Engram tables of B32's model (64 GiB rows, 48 GiB rows) with masks, configs, stats and the
# build scripts; reshard the 64 GiB table to <= 5 GB files (E043 reshard_safetensors.py, sha256-checked); upload as a
# PRIVATE HF dataset with upload_large_folder; compare the repo's file list with the staging dir. Needs $HF_TOKEN.
set -uo pipefail
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set"; exit 1; }
D=/workspace/dsv4; cd $D
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
H=$D/export/hf_engram; REPO=chankhavu/dsv41-flash1017-e066-engram-tables
rm -rf $H; mkdir -p $H/b32_tool64_mxfp4 $H/b32_tool48_mxfp4 $H/scripts
stage() {  # src_export engram_dir dest
  for f in $(ls $2 | grep -E '^model-engram-.*\.safetensors$'); do ln $2/$f $3/$f; done
  python3 - $1 $3 <<'PY'
import json, sys
src, dst = sys.argv[1:]
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
eng = {k: v for k, v in wm.items() if v.startswith("model-engram-")}
json.dump({"metadata": {"total_size": 0}, "weight_map": eng}, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
cfg = json.load(open(f"{src}/config.json")); tc = cfg.get("text_config", cfg)
json.dump({k: tc[k] for k in tc if k.startswith("engram")}, open(f"{dst}/engram_config.json", "w"), indent=1)
print(len(eng), "Engram tensors staged")
PY
}
stage export/b32_fp4 export/engram_b32_fp4 $H/b32_tool64_mxfp4
cp export/engram_b32_fp4/fp4_engram_stats.json $H/b32_tool64_mxfp4/; cp data/e066/e043tool_top64gib.npz $H/b32_tool64_mxfp4/
nice -n 10 python3 $S43/reshard_safetensors.py $H/b32_tool64_mxfp4 || { echo RESHARD_FAILED; exit 1; }
stage export/b32_48fp4 export/engram_b32_48fp4 $H/b32_tool48_mxfp4
cp export/engram_b32_48fp4/fp4_engram_stats.json export/engram_b32_48/engram_subset.json export/engram_b32_48/engram_subset_48gib_mask.npz $H/b32_tool48_mxfp4/
cp $S/cards/README_engram_tables.md $H/README.md
cp $S/fp4_engram_chunked.py $S/engram_subset.py $S/assemble_engram_variant.py $S/engram_format_error.py $S43/make_fp4_engram.py $H/scripts/
big=$(find $H -name "*.safetensors" -size +4768M | wc -l); [ "$big" = 0 ] || { echo "FILES_ABOVE_5GB $big"; exit 1; }
du -sh $H; date -u; echo "uploading -> $REPO"
nice -n 10 python3 - $H $REPO <<'PY'
import os, sys
from huggingface_hub import HfApi
path, repo = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, folder_path=path, repo_type="dataset", private=True)
info = api.dataset_info(repo, files_metadata=True)
remote = {s.rfilename: s.size for s in info.siblings}
local = {os.path.relpath(os.path.join(r, f), path): os.path.getsize(os.path.join(r, f))
         for r, _, fs in os.walk(path) for f in fs if ".cache" not in r}
bad = sorted(k for k in local if remote.get(k) != local[k])
print(f"local {len(local)} files {sum(local.values())/1e9:.2f} GB | remote {len(remote)} | mismatches {bad[:5]} ({len(bad)})")
print("ENGRAM_UPLOAD_VERIFIED" if not bad else "ENGRAM_UPLOAD_MISMATCH")
PY
date -u
