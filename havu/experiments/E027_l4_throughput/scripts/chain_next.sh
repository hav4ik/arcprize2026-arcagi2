#!/bin/bash
# L4 box, after the GGUF benches: NVFP4 decode bench, then INT4 with TurboQuant KV presets.
# Each run logs the KV capacity vLLM reports and runs bench_decode.py at 8/16/32/48 (vLLM 0.23, TP4, no prefix reuse).
cd /root/arc
trap - INT
until grep -q GGUF_DONE chain_gguf.log; do sleep 30; done
run() {  # label model kv-dtype attention-backend
  local N=$1; mkdir -p runs/$N
  MODEL=$2 ATTN=$4 VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
    --kv-cache-dtype $3 > runs/$N/server.log 2>&1 < /dev/null &
  until curl -s -m 5 localhost:8001/v1/models > /dev/null; do
    sleep 15
    if grep -q Traceback runs/$N/server.log; then
      echo "$N SERVER_FAILED: $(grep -E 'Error|error' runs/$N/server.log | grep -v Warning | tail -2 | cut -c1-300)"
      for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 20; return; fi
  done
  echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | tail -1) | $(grep -oE 'Using [A-Za-z0-9_]+ (backend|kernel)[^.]*' runs/$N/server.log | sort -u | head -3 | tr '\n' ';')"
  /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
  grep concurrency runs/$N/bench.log
  for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done
  sleep 25
}
until [ -n "$(ls /workspace/models/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/*/config.json 2>/dev/null)" ] && \
      ! pgrep -f "hf download nvidia/Qwen3.8-27B-NVFP4" > /dev/null; do sleep 30; done
run nvfp4 nvidia/Qwen3.8-27B-NVFP4 fp8 TRITON_ATTN
for kv in turboquant_k8v4 turboquant_4bit_nc turboquant_3bit_nc; do
  run int4_$kv RedHatAI/Qwen3.8-27B-INT4 $kv TURBOQUANT
done
echo NEXT_DONE
