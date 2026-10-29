#!/bin/bash
# E043 wrap-up: serving parity of the two row-chunked (<= 5 GB files) checkpoints with the row-chunk loader in a real
# vLLM server: the E112 HF export (server A) and the E128 Kaggle copy (server B). Dev NLL on the 64 tool dev traces
# vs the runtime-mask / earlier-export values (E112 runtime masks 0.8233; E128 HF export 0.79640), plus the
# first-turn probe. Run on the box.
set -u
cd /workspace/dsv4
X112=/dev/shm/e043/export/tool_e112_w10_eng64_gptq
X128=/dev/shm/e043/kaggle_e128
for s in A B; do [ $s = A ] && GP=0,1,2,3 || GP=4,5,6,7; bash code/scripts/stop.sh tp4${s}_e112t_w10_u2_gptq_eng64_sub80 $GP; done
venv/bin/python code/patches/apply_compact_engram.py venv/lib/python3.12/site-packages/vllm
grep -c "spans: dict\[str, list\[tuple\[int, int\]\]\]" venv/lib/python3.12/site-packages/vllm/models/deepseek_v4_1/nvidia/model.py
SCH13='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
for s in A B; do
  [ $s = A ] && { GP=0,1,2,3; PO=18000; M=$X112; N=parity_e112_chunked; } || { GP=4,5,6,7; PO=18001; M=$X128; N=parity_e128_kaggle; }
  echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > /dev/shm/e043/ctrl$s.json
  DSV41_EXPERT_HUMMING="$SCH13" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$M/gptq MODEL=$M GPUS=$GP TP=4 EP=0 \
    PORT=$PO CTRL=/dev/shm/e043/ctrl$s.json UTIL=0.86 SEQS=64 MBT=8192 bash code/scripts/serve.sh tp4${s}_$N
done
for N in tp4A_parity_e112_chunked tp4B_parity_e128_kaggle; do
  until grep -qE "Application startup complete|Traceback" runs/$N/server.log; do sleep 20; done
  grep -E "Application startup complete|KV cache size|Error|AssertionError|engram row chunks" runs/$N/server.log | grep -v UserWarning | tail -3 | cut -c1-220
done
date -u
for s in A B; do
  [ $s = A ] && { PO=18000; T=e112t_u2_gptq_export_chunked; } || { PO=18001; T=e128t_u2_gptq_kaggle_chunked; }
  ( SERVER=http://127.0.0.1:$PO CTRL=/dev/shm/e043/ctrl$s.json TOK=data/tok_tool_dev NSEQ=64 MAXLEN=60000 \
      bash code/scripts/nll_candidates.sh data/nll_tool_dev.jsonl $T:none:none | grep nll_gen | cut -c1-260
    tvenv/bin/python code/scripts/gen_probe.py --server http://127.0.0.1:$PO --ctrl /dev/shm/e043/ctrl$s.json \
      --parquet data/tool_dev_probe.parquet --n 8 --samples 2 --max-tokens 6000 --workers 25 \
      --out data/gen_probe_tool_dev.jsonl $T:none:none | grep '^==' ) &
done
wait
date -u; echo PARITY-CHUNKED-DONE
