#!/bin/bash
# L4 box: INT4 (RedHatAI GPTQ W4A16) decode bench with the same script and levels as the FP8 numbers
# (bench_decode.py, ~2.6k-token prompts, no prefix cache reuse, levels 8 16 32 48), vLLM 0.23 TP4, no MTP.
N=int4_fair; mkdir -p /root/arc/runs/$N
trap - INT
MODEL=RedHatAI/Qwen3.8-27B-INT4 VENV=/root/vllm-venv setsid nohup /root/arc/start_vllm.sh 0 --no-enable-prefix-caching --language-model-only > /root/arc/runs/$N/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback /root/arc/runs/$N/server.log && { echo SERVER_FAILED; exit 1; }; done
grep -o "GPU KV cache size: [0-9,]* tokens" /root/arc/runs/$N/server.log | tail -1
/opt/sglang/bin/python /root/arc/bench_decode.py http://127.0.0.1:8001 int4_fair /root/arc/runs/$N/int4_fair.json 8 16 32 48 > /root/arc/runs/$N/bench.log 2>&1
cat /root/arc/runs/$N/bench.log | grep concurrency
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server|vllm serve'); do kill $p; done
sleep 20
echo INT4_FAIR_DONE
