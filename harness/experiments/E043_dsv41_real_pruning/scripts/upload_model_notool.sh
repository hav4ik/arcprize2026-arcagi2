#!/bin/bash
# E043 no-tool round: upload one expert-pruned checkpoint *with* its own compact Engram files (the no-tool 48 GiB
# table is not shared with the tool-round exports) to a private HF model repo. Token in $HF_TOKEN.
# Usage: upload_model_notool.sh <export_dir> <repo> <E> <keep_rule_description>
set -u
cd /workspace/dsv4
X=$1; REPO=$2; E=$3; RULE=$4
S=hf_stage_$(basename $X); rm -rf $S; mkdir -p $S
for f in $X/*; do
  if [ -d $f ]; then cp -r $f $S/; else ln -f $f $S/ 2>/dev/null || cp $f $S/; fi
done
cat > $S/README.md <<MD
# DeepSeek-V4.1-Flash-1017, E${E} expert-pruned for no-tool ARC-AGI-2 (E043 no-tool round, private)

- Base: deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170.
- Routed experts: ${E} of 384 per backbone layer (top-6 routing, shared expert, attention, mHC unchanged), kept by
  ${RULE}; experts renumbered 0..${E}-1 in ascending original id (\`e043_manifest.json\` lists the original ids);
  \`ffn.gate.weight\` / \`bias\` / \`bias_vl\` sliced to the kept rows; native MXFP4 bytes + UE8M0 scales unchanged.
- Engram: compact 48 GiB table (25% of rows, top lookup frequency on no-tool calibration traffic) in this repo's
  \`model-engram-*.safetensors\`; \`text_config.engram_compact_kept\` is set. Needs the E043 fork patch
  \`apply_compact_engram.py\` (+ the router whitelist for (${E}, 6)) on the vLLM DeepSeek-V4.1 SM89/SM120 fork.
- Text only: DSpark (mtp.*) and vision tensors removed (serve with --language-model-only). The DSpark drafter
  tensors are unchanged from the base checkpoint if you want to add them back (E043 measured acceptance length ~2.5
  at N=4 for this keep set on no-tool traffic).
- Calibration: no-tool rollouts only (\`notool_e044_*\` of livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036,
  4,931 calibration rollouts, family-disjoint dev split for screening); 20M tokens replayed through the full model.
- Use with the E044 no-tool harness (official ARC Prize prompt, task summary, effort low + minimal header).
  ARC-AGI-2 eval (120 tasks / 167 pairs, runtime masks on the full checkpoint): pass@1 57.5%, pass@2 66.9%
  task-weighted (full model: pass@1 64.7%). Details: havu/experiments/E043_dsv41_real_pruning in hav4ik/arcprize2026-arcagi2.
MD
tvenv/bin/python code/scripts/hf_upload.py --repo $REPO --type model --path $S
