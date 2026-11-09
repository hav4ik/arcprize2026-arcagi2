#!/bin/bash
# vLLM 0.23 FP8 TP4 with NCCL P2P forced (NCCL_P2P_LEVEL=SYS; default NCCL picks SHM on this PHB/NODE topology).
# 1) decode + prefill benches (prefix caching off)  2) v4b probe at c24 (prefix caching on).
cd /root/arc
trap - INT
export NCCL_P2P_LEVEL=SYS
M=Qwen/Qwen3.8-27B-FP8
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
up() { until curl -s -m 5 localhost:$1/v1/models > /dev/null; do sleep 15; if grep -q Traceback $2; then echo "FAILED $2: $(grep -E 'Error' $2 | tail -1 | cut -c1-250)"; return 1; fi; done; return 0; }
N=fp8_p2p; mkdir -p runs/$N
MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only > runs/$N/server.log 2>&1 < /dev/null &
if up 8001 runs/$N/server.log; then
  echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | tail -1) | custom AR: $(grep -ioE 'custom allreduce[^.]*' runs/$N/server.log | head -1)"
  /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
  echo "prefill: $(/opt/sglang/bin/python -c "
import json; r = json.load(open('runs/$N/prefill.json'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']), 'tok/s', r['wall_s'], 's')")"
  /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
  cat runs/$N/bench.log
fi
stop
echo P2P_BENCH_DONE
./probe.sh p_fp8_p2p_c24 $M 24
echo P2P_DONE
