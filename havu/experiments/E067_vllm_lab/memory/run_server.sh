#!/usr/bin/env bash
# Start one upstream vLLM 0.20 OpenAI server on GPU 0 under the lab's GPU-0 lock,
# with the per-step memory logger (hook/sitecustomize.py) enabled.
#
#   NAME=c1 UTIL=0.90 MBT=2048 MAXLEN=65536 ./run_server.sh [extra vllm args...]
#
# Env passed through to the hook: MEMLAB_CAP_MB, MEMLAB_EXPANDABLE_AFTER,
# MEMLAB_EMPTY_CACHE_EVERY, MEMLAB_SNAP_EVERY, PYTORCH_CUDA_ALLOC_CONF.
# Writes $LOGDIR/$NAME.server.log, $LOGDIR/$NAME.<pid>.jsonl, $LOGDIR/$NAME.pid
# (the process-group id; stop with: kill -TERM -- -$(cat $LOGDIR/$NAME.pid)).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
LOGDIR="${LOGDIR:-$SCR/memory/logs}"
LOCK="$SCR/gpu0.lock"
PY=/home/vu/miniconda3/envs/vllm/bin/python
# Stand-in for DeepSeek-R1-Distill-Qwen-7B (its weights are not in the HF cache):
# an SFT of it with the identical Qwen2 7B architecture, already on disk.
MODEL="${MODEL:-/home/vu/.cache/kagglehub/models/yeoyunsianggeremie/aimo-sft-7b/Transformers/default/6}"
NAME="${NAME:?NAME}"; UTIL="${UTIL:-0.90}"; MBT="${MBT:-2048}"; MAXLEN="${MAXLEN:-65536}"
PORT="${PORT:-18467}"; MAXSEQS="${MAXSEQS:-64}"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/$NAME".*.jsonl
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$HERE/hook"
export MEMLAB_LOG="$LOGDIR/$NAME"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export HF_HUB_OFFLINE=1
setsid flock "$LOCK" "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name m --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MAXSEQS" \
  "$@" > "$LOGDIR/$NAME.server.log" 2>&1 &
PG=$!
echo "$PG" > "$LOGDIR/$NAME.pid"
echo "started pgid $PG"
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ready after $((i * 2)) s"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i 0 | tee "$LOGDIR/$NAME.nvsmi_ready.txt"
    exit 0
  fi
  if ! kill -0 "$PG" 2>/dev/null; then echo "server exited"; tail -30 "$LOGDIR/$NAME.server.log"; exit 1; fi
  sleep 2
done
echo "timeout"; exit 1
