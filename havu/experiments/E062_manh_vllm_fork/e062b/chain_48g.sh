#!/bin/bash
# E062b on the 48 GB 4090 box (58066982): Manh's recommended flags.
# After acc_r1's chain is done:
#   1. teacher-forced numerics (E062's 24 prompts, B32's continuations) on a standalone fork server with Manh's flags;
#   2. acc_manhflags: acc_manh3's exact podrun env (fork, B32 kit, x56, 120 tasks x 1) + Manh's flags in the server env;
#   3. the same teacher-forced probe without Manh's flags (acc_manh3's server env) as the same-box reference.
set -uo pipefail
D=/root/e062b; L=$D/chain.log; cd $D
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
gpus_free() { for _ in $(seq 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1000 ] && return 0; sleep 10; done; return 1; }
FLAGS="VLLM_HOST_ALLREDUCE=1 VLLM_HOST_ALLREDUCE_MODE=60 VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=16 VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB=36 VLLM_HOST_ALLREDUCE_LARGE_SHAPE=32x32x256 VLLM_HOST_ALLREDUCE_SHAPE=12x12x256 VLLM_HOST_ALLREDUCE_UNIT_KB=16 VLLM_DCP_ASYNC_QUERY_GATHER=1 VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 VLLM_TRITON_SPARSE_INDEXER_LOGITS=1 VLLM_GPU_DRIVER_RESERVE_MB=512 VLLM_GPU_MEMORY_LOG_INTERVAL=60"
BASE="VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 VLLM_GPU_DRIVER_RESERVE_MB=512 VLLM_GPU_MEMORY_LOG_INTERVAL=300"
OPTS=(--linear-backend humming --cp-kv-cache-interleave-size 64 --cpu-offload-gb 1 --cpu-offload-params embed_tokens)
tf_probe() {  # $1 tag, $2 env string
  local tag=$1 port=18300
  E062_MODEL=/root/dsv41/model-e128t E062_UTIL=0.92 E062_ENV="$2" bash $D/serve.sh manh $port "${OPTS[@]}" >> $L 2>&1
  local pid; pid=$(cat /root/e062/serve_manh_${port}.pid)
  for _ in $(seq 180); do
    curl -sf http://127.0.0.1:$port/v1/models > /dev/null && break
    kill -0 $pid 2>/dev/null || { log "$tag server died"; tail -30 /root/e062/serve_manh_${port}.log >> $L; return 1; }
    sleep 10
  done
  curl -sf http://127.0.0.1:$port/v1/models > /dev/null || { log "$tag server not up after 30 min"; kill -- -$pid; return 1; }
  log "$tag server up: $(grep -m1 -o 'GPU KV cache size: [0-9,]* tokens' /root/e062/serve_manh_${port}.log)"
  cp /root/e062/serve_manh_${port}.log $D/serve_$tag.log
  python3 -I $D/parity.py tf --prompts $D/prompts.json --ref $D/run_b32_ref.json --url http://127.0.0.1:$port \
    --out $D/tf_$tag.json --salt e062b_$tag > $D/tf_$tag.log 2>&1 && log "$tag tf done" || log "$tag tf FAILED"
  cp /root/e062/serve_manh_${port}.log $D/serve_$tag.log
  kill -- -$pid 2>/dev/null; sleep 20; gpus_free || log "GPUs not free after $tag"
}
until grep -q "ACC_R1 CHAIN DONE" /root/e062/acc_r1_chain.log 2>/dev/null; do sleep 60; done
log "acc_r1 chain done"; gpus_free || { log "GPU busy, abort"; exit 1; }
tf_probe manhflags "$FLAGS"
ARGS=()
while IFS= read -r line; do
  case "${line%%=*}" in ARC_OUT|ARC_SCRATCH|ARC_CACHE_SALT|ARC_SERVER_ENV_JSON) continue ;; esac
  ARGS+=("$line")
done < <(grep -E '^ARC_' /root/runs/acc_manh3/env.txt)
SENV=$(python3 -I -c 'import json,sys; print(json.dumps(dict(kv.split("=",1) for kv in sys.argv[1].split())))' "$FLAGS")
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HF_API_TOKEN$)'); do unset "$v"; done
bash /root/podrun.sh acc_manhflags "${ARGS[@]}" "ARC_SERVER_ENV_JSON=$SENV" \
  PATH=/root/manh/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin >> $L 2>&1
sleep 5; setsid nohup /root/memlog.sh /root/runs/acc_manhflags/mem.log > /dev/null 2>&1 < /dev/null &
M=$!; log "acc_manhflags launched (memlog $M); env: $(grep ARC_SERVER_ENV_JSON /root/runs/acc_manhflags/env.txt)"
until grep -q "^done at +" /root/runs/acc_manhflags/run.log 2>/dev/null; do sleep 60; done
log "acc_manhflags done: $(grep 'local score' /root/runs/acc_manhflags/out/dsv41_logs/solver.log | tail -1)"
kill $M 2>/dev/null; gpus_free || log "GPUs not free after acc_manhflags"
tf_probe base "$BASE"
log "E062B 48G CHAIN DONE"
