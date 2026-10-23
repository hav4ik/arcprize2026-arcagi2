#!/bin/bash
# E043: upload one expert-pruned checkpoint (backbone files only; the compact Engram files are shared and live in
# chankhavu/DeepSeek-V4.1-Flash-1017-arc-engram64) to a private HF model repo. Token in $HF_TOKEN.
# Usage: upload_model.sh <export_dir> <repo> <E> <keep_rule_description>
set -u
cd /workspace/dsv4
X=$1; REPO=$2; E=$3; RULE=$4
S=hf_stage_$(basename $X); rm -rf $S; mkdir -p $S
for f in $X/*; do
  b=$(basename $f)
  case $b in model-engram-*) continue;; esac
  if [ -d $f ]; then cp -r $f $S/; else ln -f $f $S/ 2>/dev/null || cp $f $S/; fi
done
cat > $S/README.md <<MD
# DeepSeek-V4.1-Flash-1017, E${E} expert-pruned for ARC-AGI-2 (E043, private)

- Base: deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170.
- Routed experts: ${E} of 384 per backbone layer (top-6 routing, shared expert, attention, mHC unchanged), kept by
  ${RULE}; experts renumbered 0..${E}-1 in ascending original id (\`e043_manifest.json\` lists the original ids);
  \`ffn.gate.weight\` / \`bias\` / \`bias_vl\` sliced to the kept rows; native MXFP4 bytes + UE8M0 scales unchanged.
- Engram: compact 64 GiB table (34% of rows) in the separate repo chankhavu/DeepSeek-V4.1-Flash-1017-arc-engram64.
  Download its \`model-engram-*.safetensors\` into this folder: \`model.safetensors.index.json\` already maps them,
  and \`text_config.engram_compact_kept\` is set. Needs the E043 fork patch \`apply_compact_engram.py\`
  (+ the router whitelist for (${E}, 6)) on the vLLM DeepSeek-V4.1 SM89/SM120 fork.
- Text only: DSpark (mtp.*) and vision tensors removed (serve with --language-model-only, no speculative decoding).
- Calibration: 10.0M tokens of E031/E036/E038 ARC tool-use rollouts replayed through the full model (REAP-style
  native-amplitude statistics). Quality and evals: see havu/experiments/E043_dsv41_real_pruning in
  hav4ik/arcprize2026-arcagi2.
MD
tvenv/bin/python code/scripts/hf_upload.py --repo $REPO --type model --path $S
echo MODEL_UPLOAD_DONE $REPO
