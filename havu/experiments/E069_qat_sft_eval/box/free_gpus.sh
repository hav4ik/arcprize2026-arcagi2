#!/bin/bash
# E068 box: stop the image's template vLLM (supervisor) and kill leftover GPU processes by PID only.
supervisorctl stop vllm
sleep 10
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  echo "killing leftover GPU pid $p: $(ps -o cmd= -p $p | cut -c1-120)"; kill $p
done
sleep 10
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv
