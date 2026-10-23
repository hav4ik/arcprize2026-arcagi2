#!/bin/bash
# E043: serve the full DeepSeek-V4.1-Flash-1017 (all 384 experts, full Engram table in pinned host memory)
# on the 8x RTX PRO 6000 (SM120) box with the E043 hooks (runtime expert/Engram masks, REAP observer).
# Usage: serve.sh <run_name> [extra vllm args...]
#   env: PORT (18000), TP (8), EP (1 = --enable-expert-parallel), SEQS (64), MBT (8192), UTIL (0.90),
#        MAXLEN (131072), EAGER (0), PREFIX (1), MODEL, CTRL (/dev/shm/e043/ctrl.json), GPUS (0-7)
set -u
NAME=$1; shift
R=/workspace/dsv4/runs/$NAME; mkdir -p $R /dev/shm/e043; rm -f $R/server.pid
V=/workspace/dsv4/venv
export PATH=$V/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=12.0 TORCH_CUDA_ARCH_LIST=12.0
export TRITON_PTXAS_BLACKWELL_PATH=$V/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell
unset VLLM_MARLIN_INPUT_DTYPE NCCL_P2P_LEVEL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export E043_CTRL=${CTRL:-/dev/shm/e043/ctrl.json}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}
GRAPH=(--compilation-config '{"cudagraph_mode":"FULL"}')
[ "${EAGER:-0}" = 1 ] && GRAPH=(--enforce-eager)
EPF=(); [ "${EP:-1}" = 1 ] && EPF=(--enable-expert-parallel)
PC=(--enable-prefix-caching); [ "${PREFIX:-1}" = 0 ] && PC=(--no-enable-prefix-caching)
cmd=(vllm serve ${MODEL:-/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017} --served-model-name dsv41
  --host 127.0.0.1 --port ${PORT:-18000} --trust-remote-code
  --tensor-parallel-size ${TP:-8} "${EPF[@]}" --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len ${MAXLEN:-131072}
  --max-num-seqs ${SEQS:-64} --max-num-batched-tokens ${MBT:-8192} --gpu-memory-utilization ${UTIL:-0.90}
  "${PC[@]}" --engram-config '{"cpu_offload":true}'
  --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  "${GRAPH[@]}" "$@")
printf '%q ' "${cmd[@]}" > $R/cmd.txt; echo >> $R/cmd.txt
env | grep -E '^(VLLM_|E043_|FLASHINFER_|TORCH_CUDA|CUDA_|HF_HUB|NCCL|TRITON)' | sort > $R/env.txt
cd $R
setsid bash -c 'echo $$ > server.pid; exec "$@"' _ "${cmd[@]}" > $R/server.log 2>&1 < /dev/null &
for i in 1 2 3 4 5; do [ -s $R/server.pid ] && break; sleep 1; done
echo "started $NAME pid/sid $(cat $R/server.pid)"
