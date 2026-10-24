#!/bin/bash
# E042: serve the synthetic E64 DeepSeek-V4.1-Flash (random, format-valid weights) on 4xL4.
# Usage: serve.sh <run_name> [extra vllm args...]   (env: PORT, SEQS, MBT, UTIL, MAXLEN, EAGER, ENGRAM_ROWS, MODEL)
# Starts in the background; logs to /root/dsv41/runs/<run_name>/server.log; writes server.pid.
set -u
NAME=$1; shift
R=/root/dsv41/runs/$NAME; mkdir -p $R; rm -f $R/server.pid
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:/root/dsv41/venv/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9
unset VLLM_MARLIN_INPUT_DTYPE TRITON_PTXAS_BLACKWELL_PATH NCCL_P2P_LEVEL
[ -n "${P2P:-}" ] && export NCCL_P2P_LEVEL=$P2P   # box-only experiments; never on Kaggle
[ -n "${MARLIN_IN:-}" ] && export VLLM_MARLIN_INPUT_DTYPE=$MARLIN_IN   # W4A8 Marlin experts (fp8 | int8)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DSV41_SYNTH_INIT=1 DSV41_SYNTH_MANIFEST=$R/synth
export DSV41_ENGRAM_SYNTH_ROWS=${ENGRAM_ROWS:-4000000}   # 4M rows x 264 B = 0.98 GiB per rank and layer, 7.9 GiB total
GRAPH=(--compilation-config '{"cudagraph_mode":"FULL"}')
[ "${EAGER:-0}" = 1 ] && GRAPH=(--enforce-eager)
cmd=(vllm serve ${MODEL:-/root/dsv41/model-e64} --served-model-name dsv41-e64 --host 127.0.0.1 --port ${PORT:-8100}
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len ${MAXLEN:-65536}
  --max-num-seqs ${SEQS:-8} --max-num-batched-tokens ${MBT:-2048} --gpu-memory-utilization ${UTIL:-0.85}
  --no-enable-prefix-caching --engram-config '{"cpu_offload":true}'
  --load-format dummy --tokenizer-mode deepseek_v41 --language-model-only
  "${GRAPH[@]}" "$@")
printf '%q ' "${cmd[@]}" > $R/cmd.txt; echo >> $R/cmd.txt
env | grep -E '^(VLLM_|DSV41_|FLASHINFER_|TORCH_CUDA|CUDA_HOME|HF_HUB|NCCL)' | sort > $R/env.txt
# The session leader writes its own pid (= session id) and execs vllm; stop.sh kills that session.
cd $R
setsid bash -c 'echo $$ > server.pid; exec "$@"' _ "${cmd[@]}" > $R/server.log 2>&1 < /dev/null &
for i in 1 2 3 4 5; do [ -s $R/server.pid ] && break; sleep 1; done
echo "started $NAME pid/sid $(cat $R/server.pid)"
