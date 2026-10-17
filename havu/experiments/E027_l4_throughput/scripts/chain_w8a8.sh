#!/bin/bash
# FP8 track (2026-10-16 ~20:05 UTC): official Qwen FP8 checkpoint (128x128 block FP8, activation_scheme dynamic).
#  a) fp8_marlin:   default kernel on L4 (MarlinFP8ScaledMMLinearKernel, weight-only, bf16 activations) = v4b
#  b) fp8_triton:   Marlin and Humming disabled -> block-FP8 W8A8 (FP8 activations, FP8 tensor cores on Ada)
#  c) best of a/b with --async-scheduling
# Each: decode bench (bench_decode.py 8/16/32/48) + prefill bench (bench_decode2.py, ~12k-token prompts, 16 out, c16).
cd /root/arc
trap - INT
for p in $(pgrep -f 'chain_prio.sh'); do kill $p; done
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done
sleep 25
rm -rf /workspace/models/models--cyankiwi--Qwen3.8-27B-AWQ-INT4 /workspace/gguf/bonsai
df -h / | tail -1
/root/vllm-venv/bin/hf download Qwen/Qwen3.8-27B-FP8 --cache-dir /workspace/models > logs/dl_fp8b.log 2>&1
echo "download done $(date -u +%H:%M) $(du -sh /workspace/models/models--Qwen--Qwen3.8-27B-FP8 | cut -f1)"
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 25; }
up() { until curl -s -m 5 localhost:$1/v1/models > /dev/null; do sleep 15; if grep -q Traceback $2; then echo "FAILED $2: $(grep -E 'Error' $2 | tail -1 | cut -c1-250)"; return 1; fi; done; return 0; }
run() {  # label env-assignments... -- extra vLLM flags
  local N=$1; shift; local ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
  mkdir -p runs/$N
  env "${ENVS[@]}" MODEL=Qwen/Qwen3.8-27B-FP8 VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching \
    --language-model-only "$@" > runs/$N/server.log 2>&1 < /dev/null &
  if up 8001 runs/$N/server.log; then
    echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | tail -1) | $(grep -oE 'Using [A-Za-z0-9]+(Kernel|kernel)' runs/$N/server.log | sort -u | tr '\n' ' ')"
    /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
    cat runs/$N/bench.log
    /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
    echo "prefill: $(cat runs/$N/prefill.log)"
  fi
  stop
}
run fp8_marlin X=1 --
run fp8_triton VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel --
best=$(/opt/sglang/bin/python -c "
import json
s = {n: sum(r['tok_s'] for r in json.load(open(f'runs/{n}/{n}.json'))) for n in ('fp8_marlin', 'fp8_triton')}
print(max(s, key=s.get))" 2>/dev/null || echo fp8_marlin)
echo "best decode: $best"
if [ "$best" = fp8_triton ]; then
  run fp8_triton_async VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel -- --async-scheduling
else
  run fp8_marlin_async X=1 -- --async-scheduling
fi
echo W8A8_DONE
