#!/bin/bash
# Start the E032 server with the Kaggle notebook's flags only (no tool/reasoning parsers): TP4, official tokenizer, --language-model-only.
M=$1; R=/root/arc/runs/$2; mkdir -p $R; cd /root/arc
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 20
for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5
(MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --language-model-only --tokenizer Qwen/Qwen3.8-27B-FP8 > $R/server.log 2>&1 < /dev/null &)
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback $R/server.log && { echo SERVE_FAILED; exit 1; }; done
echo "up $2 $(date -u +%H:%M): $(grep -oE 'Selected [A-Za-z0-9]+ for [A-Za-z0-9]+' $R/server.log | sort -u | tr '\n' ' ')"
ps -o args= -p $(pgrep -f "vllm.entrypoints.openai.api_server" | head -1) | tr ' ' '\n' | grep -cE "parser|tool-choice"
