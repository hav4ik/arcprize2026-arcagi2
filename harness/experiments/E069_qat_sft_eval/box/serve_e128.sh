#!/bin/bash
# E068: serve tool E128 2-bit (HF chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit, B32's model) on the
# 2x RTX PRO 6000 Blackwell Server box (sm120) at TP2 on GPUs 0,1.
# Stack: the E043 fork venv (locks/venv_vllm.freeze) + E043 patches (observer/masks with a null control file, compact
# Engram, Humming 2-bit experts), i.e. E066's B32 Engram A/B stack. Flags: B32's solver_dsv41.py server_cmd() + the B32
# notebook's ARC_SERVER_EXTRA_ARGS where they exist in this stack, with E043 serve.sh's sm120 env.
# Not used (Kaggle/L4 only or not in this stack): KV CPU offload, util 0.945, --linear-backend humming (needs the kit
# overlay; would fall back), DSV41_INDEXER_Q_PREDECODE (kit overlay #154), #156 indexer skip.
# Usage: serve_e128.sh <run_name>   env: MODEL, PORT (18000), GPUS (0,1), TP (2), SEQS (64), MBT (2048),
#        UTIL (0.90), MAXLEN (163840), RETAIN (512; 0 = flag off), NAME_SERVED (dsv41)
set -u
NAME=$1; shift
D=/root/e068; R=$D/runs/$NAME; mkdir -p $R $D/cache /dev/shm/e043; rm -f $R/server.pid
V=$D/venv; SP=$V/lib/python3.12/site-packages; M=${MODEL:-$D/model-e128t}
unset VLLM_ARGS VLLM_MODEL VLLM_TEST_ENDPOINT VLLM_USAGE_SOURCE VLLM_IMAGE_TAG VLLM_BUILD_URL VLLM_BUILD_PIPELINE \
  VLLM_BUILD_COMMIT VLLM_ENABLE_CUDA_COMPATIBILITY VLLM_MARLIN_INPUT_DTYPE NCCL_P2P_LEVEL HF_TOKEN
export PATH=$V/bin:/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=12.0 TORCH_CUDA_ARCH_LIST=12.0
export TRITON_PTXAS_BLACKWELL_PATH=$SP/triton/backends/nvidia/bin/ptxas-blackwell
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=$D/cache/vllm FLASHINFER_WORKSPACE_BASE=$D/cache
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR=$M/gptq
# E043 hooks: a null control file = the physically pruned model as shipped (check "ctrl loaded: experts=None engram=None")
export E043_CTRL=/dev/shm/e043/ctrl_$NAME.json
echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $E043_CTRL
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
RET=(); [ "${RETAIN:-512}" != 0 ] && RET=(--prefix-cache-retention-interval ${RETAIN:-512})
cmd=(vllm serve $M --served-model-name ${NAME_SERVED:-dsv41} --host 127.0.0.1 --port ${PORT:-18000} --trust-remote-code
  --tensor-parallel-size ${TP:-2} --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len ${MAXLEN:-163840}
  --max-num-seqs ${SEQS:-64} --max-num-batched-tokens ${MBT:-2048} --gpu-memory-utilization ${UTIL:-0.90}
  --enable-prefix-caching "${RET[@]}" --engram-config '{"cpu_offload":true}'
  --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":64}' "$@")
printf '%q ' "${cmd[@]}" > $R/cmd.txt; echo >> $R/cmd.txt
env | grep -E '^(VLLM_|E043_|DSV41_|FLASHINFER_|TORCH_CUDA|CUDA_|HF_HUB|NCCL|TRITON|LD_)' | sort > $R/env.txt
cd $R
setsid bash -c 'echo $$ > server.pid; exec "$@"' _ "${cmd[@]}" > $R/server.log 2>&1 < /dev/null &
for i in 1 2 3 4 5; do [ -s $R/server.pid ] && break; sleep 1; done
echo "started $NAME pid/sid $(cat $R/server.pid)"
