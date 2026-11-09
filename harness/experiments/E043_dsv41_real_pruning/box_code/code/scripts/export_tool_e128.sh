#!/bin/bash
# E043 tool round: GPTQ-only export of tool-use E128 2-bit (keep_tool w10, 64 GiB tool Engram hardlinked from e72_w30_eng64).
set -e
cd /workspace/dsv4
K=data/reap/keep_tool/keep_contrib_dec_sp0_w10_e128.npy
venv/bin/python code/scripts/export_pruned.py --src /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017 --expert-keep $K \
  --engram-from export/e72_w30_eng64 --text-only --no-routed-experts --out export/tool_e128_w10_eng64_base
venv/bin/python code/scripts/make_lowbit_export.py --base export/tool_e128_w10_eng64_base \
  --precomputed /dev/shm/e043/gptq_tool_e128_u2 --precomputed-ids original --keep $K \
  --schema "{\"quant_method\": \"humming\", \"weight_dtype\": \"uint2\", \"weight_scale_dtype\": \"bfloat16\", \"group_size\": 128, \"has_zero_point\": true}" \
  --schema-w2 "{\"group_size\": 64}" --name "E128 2-bit GPTQ, tool use" \
  --keep-desc "tool-use keep set: contribution score on tool rollouts, tool-call-block weight 10" \
  --out export/tool_e128_w10_eng64_gptq
cp -r /dev/shm/e043/gptq_tool_e128_u2/*.json export/tool_e128_w10_eng64_gptq/gptq/ 2>/dev/null || true
du -sb export/tool_e128_w10_eng64_gptq; du -sh --apparent-size export/tool_e128_w10_eng64_gptq; df -h /workspace | tail -1
echo EXPORT-TOOL-E128-DONE
