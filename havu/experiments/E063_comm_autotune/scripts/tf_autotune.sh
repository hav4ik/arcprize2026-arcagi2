#!/bin/bash
# E063: teacher-forced parity of the autotune server against E062's fork (L4 options) and B32 runs.
set -u
cd /root/e063
PORT=18300
export E062_MODEL=/root/dsv41/model-e128k
E062_ENV="VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 VLLM_GPU_DRIVER_RESERVE_MB=512 VLLM_COMM_AUTOTUNE=1" \
  bash /root/e062/serve.sh manh $PORT --linear-backend humming --cp-kv-cache-interleave-size 64 \
  --cpu-offload-gb 1 --cpu-offload-params embed_tokens
pidf=/root/e062/serve_manh_${PORT}.pid
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; kill -0 $(cat $pidf) 2>/dev/null || { echo "server died"; exit 1; }; sleep 10; done
echo "[$(date -u +%H:%M:%S)] up"
P=/root/manh/venv/bin/python
cd /root/e062
$P parity2.py tf --prompts prompts.json --ref run_b32_ref.json --url http://127.0.0.1:$PORT --out /root/e063/tf_manh_autotune.json > /root/e063/par_tf_autotune.log 2>&1
echo "fork L4opt (default comm) vs fork autotune:"; $P parity2.py tfcompare tf_manh_l4opt.json /root/e063/tf_manh_autotune.json | tail -1
echo "B32 run 1 vs fork autotune:"; $P parity2.py tfcompare tf_b32_1.json /root/e063/tf_manh_autotune.json | tail -1
echo "B32 run 1 vs B32 run 2 (noise floor):"; $P parity2.py tfcompare tf_b32_1.json tf_b32_2.json | tail -1
echo "B32 run 1 vs fork L4opt:"; $P parity2.py tfcompare tf_b32_1.json tf_manh_l4opt.json | tail -1
kill -- -$(cat $pidf) 2>/dev/null; sleep 20; kill -9 -- -$(cat $pidf) 2>/dev/null
echo TF_DONE
