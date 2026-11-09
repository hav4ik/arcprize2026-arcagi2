#!/bin/bash
# Restart the E032 server with per-request cache details and run the idle and loaded cache probes.
cd /root/arc; R=runs/e034_cache; mkdir -p $R e032_traces
cp /root/arc/work_sft/arc_logs/trace_*.txt /root/arc/e032_traces/; ls e032_traces | wc -l
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 20
for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5
(MODEL=/workspace/models/e032_sft_fp8block VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --language-model-only --tokenizer Qwen/Qwen3.8-27B-FP8 \
  --enable-prompt-tokens-details > $R/server.log 2>&1 < /dev/null &)
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback $R/server.log && { echo SERVE_FAILED; tail -3 $R/server.log; exit 1; }; done
echo "server up $(date -u +%H:%M) $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
/opt/sglang/bin/python /root/arc/cache_diag.py idle $R/idle.json > $R/idle.log 2>&1; tail -13 $R/idle.log
/opt/sglang/bin/python /root/arc/cache_diag.py load $R/load.json > $R/load.log 2>&1; tail -13 $R/load.log
grep -iE "preempt|evict" $R/server.log | tail -3
echo CACHE_DIAG_DONE
