#!/bin/bash
# E046: serve tool E128 2-bit (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit) on 4x RTX PRO 6000 (sm120),
# TP4. Kaggle's E128 command (solver_dsv41.py server_cmd/server_env) with E043 serve.sh's sm120 env; no L4-only knobs.
# Usage: serve_e128.sh <run_name>   env: SEQS (128), MBT (8192), UTIL (0.90), MAXLEN (163840)
set -u
NAME=$1; shift
R=/root/dsv41/runs/$NAME; mkdir -p $R /root/dsv41/cache; rm -f $R/server.pid
V=/root/dsv41/venv; SP=$V/lib/python3.12/site-packages; M=/root/dsv41/model-e128t
unset VLLM_ARGS VLLM_MODEL VLLM_TEST_ENDPOINT VLLM_USAGE_SOURCE VLLM_IMAGE_TAG VLLM_BUILD_URL VLLM_BUILD_PIPELINE \
  VLLM_BUILD_COMMIT VLLM_ENABLE_CUDA_COMPATIBILITY VLLM_MARLIN_INPUT_DTYPE NCCL_P2P_LEVEL
export PATH=$V/bin:/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=12.0 TORCH_CUDA_ARCH_LIST=12.0
export TRITON_PTXAS_BLACKWELL_PATH=$SP/triton/backends/nvidia/bin/ptxas-blackwell
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=/root/dsv41/cache/vllm FLASHINFER_WORKSPACE_BASE=/root/dsv41/cache
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR=$M/gptq
export CUDA_VISIBLE_DEVICES=0,1,2,3
cmd=(vllm serve $M --served-model-name dsv41 --host 127.0.0.1 --port 18000 --trust-remote-code
  --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len ${MAXLEN:-163840}
  --max-num-seqs ${SEQS:-128} --max-num-batched-tokens ${MBT:-8192} --gpu-memory-utilization ${UTIL:-0.90}
  --enable-prefix-caching --prefix-cache-retention-interval 128 --engram-config '{"cpu_offload":true}'
  --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --compilation-config '{"cudagraph_mode":"FULL"}' "$@")
printf '%q ' "${cmd[@]}" > $R/cmd.txt; echo >> $R/cmd.txt
env | grep -E '^(VLLM_|E043_|DSV41_|FLASHINFER_|TORCH_CUDA|CUDA_|HF_HUB|NCCL|TRITON|LD_)' | sort > $R/env.txt
cd $R
setsid bash -c 'echo $$ > server.pid; exec "$@"' _ "${cmd[@]}" > $R/server.log 2>&1 < /dev/null &
for i in 1 2 3 4 5; do [ -s $R/server.pid ] && break; sleep 1; done
echo "started $NAME pid/sid $(cat $R/server.pid)"
