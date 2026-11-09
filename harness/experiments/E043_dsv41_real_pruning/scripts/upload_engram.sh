#!/bin/bash
# Upload the shared compact Engram files (64 GiB mask) as their own private model repo.
set -u
cd /workspace/dsv4
X=export/e64_oneshot_contribdec_eng64
S=hf_engram64; rm -rf $S; mkdir -p $S
for f in $X/model-engram-*.safetensors; do ln -f $f $S/; done
cp data/engram/masks/engram_top64gib.json $S/mask_manifest.json
python3 - <<PY
import json
idx = json.load(open("$X/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open("$X/config.json"))["text_config"]
json.dump({"weight_map": {k: v for k, v in idx.items() if v.startswith("model-engram-")},
           "engram_compact_kept": cfg["engram_compact_kept"]}, open("$S/engram_index.json", "w"), indent=1)
PY
cat > $S/README.md <<'MD'
# DeepSeek-V4.1-Flash-1017 compact Engram table, 64 GiB (E043, private)

Top-frequency Engram rows (global ranking over both layers) from a 251.8M-token census of the E031/E036/E038 ARC
tool-use rollouts: 136,114,884 of 384,006,168 rows (layer 1) and 124,186,164 of 384,016,682 (layer 14), original
FP8 bytes + UE8M0 scales unchanged, plus `layers.{1,14}.engram.embed_tokens.row_map` (int32, original hashed row id ->
compact row, -1 = dropped = zero row before Wkv). `engram_index.json` has the weight map and the
`engram_compact_kept` counts (24 hash columns per layer) that go into `text_config` of the pruned checkpoints.
Shared by the E48/E64/E72 expert-pruned checkpoints; load with the E043 `apply_compact_engram.py` fork patch.
Held-out eval-trace NLL vs the full table: +0.09% (within run-to-run noise); lookup hit rate 94.0%.
MD
tvenv/bin/python code/scripts/hf_upload.py --repo chankhavu/DeepSeek-V4.1-Flash-1017-arc-engram64 --type model --path $S
echo ENGRAM_UPLOAD_DONE
