#!/bin/bash
# Prefill throughput vs --max-num-batched-tokens (vLLM default on <70 GB GPUs = 2048, which v4b uses),
# official FP8 checkpoint, Marlin, TRITON_ATTN, TP4. Then decode bench and a v4b probe (c24) with the best value.
cd /root/arc
trap - INT
M=Qwen/Qwen3.8-27B-FP8
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 25; }
up() { until curl -s -m 5 localhost:$1/v1/models > /dev/null; do sleep 15; if grep -q Traceback $2; then echo "FAILED $2: $(grep -E 'Error' $2 | tail -1 | cut -c1-250)"; return 1; fi; done; return 0; }
prefill_tps() { /opt/sglang/bin/python -c "
import json; r = json.load(open('$1'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']))"; }
for B in 4096 8192 16384; do
  N=fp8_mnbt$B; mkdir -p runs/$N
  MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
    --max-num-batched-tokens $B > runs/$N/server.log 2>&1 < /dev/null &
  if up 8001 runs/$N/server.log; then
    /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
    echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | tail -1) prefill $(prefill_tps runs/$N/prefill.json) tok/s ($(cat runs/$N/prefill.log | cut -c1-160))"
  fi
  stop
done
echo "baseline 2048: prefill $(prefill_tps runs/fp8_marlin/prefill.json) tok/s"
best=$(/opt/sglang/bin/python -c "
import json
s = {}
for b in (4096, 8192, 16384):
    try:
        r = json.load(open(f'runs/fp8_mnbt{b}/prefill.json'))[0]; s[b] = r['concurrency'] * r['prompt_tok'] / r['wall_s']
    except Exception: pass
print(max(s, key=s.get) if s else 2048)")
echo "best mnbt: $best"
N=fp8_mnbt${best}_decode; mkdir -p runs/$N
MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
  --max-num-batched-tokens $best > runs/$N/server.log 2>&1 < /dev/null &
if up 8001 runs/$N/server.log; then
  /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
  cat runs/$N/bench.log
fi
stop
echo MNBT_BENCH_DONE
./probe.sh p_fp8_mnbt${best}_c24 $M 24 --max-num-batched-tokens $best
echo MNBT_DONE
