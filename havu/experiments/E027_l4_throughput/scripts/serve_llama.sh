#!/bin/bash
# usage: serve_llama.sh <llama build dir> <model.gguf> <parallel slots per GPU> <total ctx per GPU> <kv type> <label> [extra flags...]
# One llama-server per GPU on ports 8101-8104 (data parallel), slots share one KV pool (--kv-unified).
B=$1; M=$2; NP=$3; C=$4; KV=$5; L=$6; shift 6
mkdir -p /root/arc/runs/llama
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup $B/bin/llama-server -m $M -ngl 99 -fa on -c $C -np $NP --kv-unified \
    -ctk $KV -ctv $KV --host 127.0.0.1 --port $((8101 + g)) --no-webui --metrics "$@" \
    > /root/arc/runs/llama/${L}_gpu$g.log 2>&1 < /dev/null &
done
for g in 0 1 2 3; do
  for i in $(seq 1 120); do
    curl -s -m 5 localhost:$((8101 + g))/health | grep -q ok && break
    sleep 5
    if ! pgrep -f "llama-server.*--port $((8101 + g))" > /dev/null; then echo "gpu$g FAILED"; tail -5 /root/arc/runs/llama/${L}_gpu$g.log; break; fi
  done
done
for g in 0 1 2 3; do echo "gpu$g $(curl -s -m 5 localhost:$((8101 + g))/health) $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader)"; done
echo SERVE_DONE
