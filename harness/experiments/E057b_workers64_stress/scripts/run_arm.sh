#!/usr/bin/env bash
# E057b: one arm of the B+ worker-count stress test on the 4x4090 box: the real solver through /root/podrun.sh +
# /root/local_run.py (as the L4 replicas, E055 replica_run.sh / launch_candBp_x2t.sh), B+ exactly (pinned_candb kit,
# venv_candbp = candidate B + E051 overlay v2, gate 0.85 + queue hold, 16 GiB offload) with the KV pool pinned to
# Kaggle's (--num-gpu-blocks-override 21490 at util 0.92), on the 90 heaviest evalx2t_s2026 tasks, ~80-min deadlines.
# Samplers: sampler.py (10 s: /metrics, status.json, host memory), memdetail (30 s, incl. the server group's PSS),
# nvidia-smi (30 s). Memory guard (coordinator, 2026-10-27: the server may use at most ~55% of the box): when the
# server process group's Pss_Anon + Pss_Shmem exceeds 55% of MemTotal in 2 samples in a row, the solver's process group
# gets SIGTERM (it writes its submission and stops its server). Waits for the run to end, then checks that the server's
# process group is gone (stops it by PGID if not) and the GPUs are idle.
#
#   bash run_arm.sh <name> <workers> [offload bytes, default 17179869184 = 16 GiB]
#   (progress: /root/e057b/runs/<name>.progress)
set -uo pipefail
N=${1:?name}; W=${2:?workers}; OFF=${3:-17179869184}
[[ "$OFF" =~ ^[0-9]+$ ]] || { echo "bad offload bytes"; exit 2; }
R=/root/runs/$N
P=/root/e057b/runs/$N.progress
mkdir -p /root/e057b/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$P"; }
[ ! -e "$R" ] || { log "$R exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { log "a GPU holds ${used} MiB: not starting"; exit 1; }
if pgrep -f 'python[0-9.]* -u [^ ]*/[l]ocal_run[.]py|python[0-9.]* -u [^ ]*/[s]olver_dsv41[.]py' > /dev/null; then
  log "a solver is already running: not starting"; exit 1; fi
IDS=$(tr -d ' \n' < /root/e057b/workloads/evalx2t_s2026_heavy90_ids.txt)
EXTRA='["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":'"$OFF"',\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}", "--num-gpu-blocks-override", "21490"]'
SALT=${N}_$(date -u +%H%M%S)
# The image's environment carries credentials and its own Qwen vLLM settings; none of them belong in the solver,
# the server or the tool sandbox (Kaggle has none of them).
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HUGGINGFACE_HUB_TOKEN$|HF_API_TOKEN$|WANDB_API_KEY$|CONTAINER_API_KEY$|JUPYTER_TOKEN$|OPEN_BUTTON_TOKEN$)'); do
  unset "$v"; done
MT_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
GUARD_PCT=${GUARD_PCT:-55}  # the server may use at most this % of MemTotal (an approved 32 GiB arm C may set more)
GUARD_KB=$(( MT_KB * GUARD_PCT / 100 ))
log "start $N: workers $W, offload $OFF B, MemTotal $(( MT_KB / 1048576 )) GiB, cgroup max $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null), guard $(( GUARD_KB / 1048576 )) GiB, salt $SALT, $(echo "$IDS" | tr ',' '\n' | wc -l) task ids; host $(awk '/^(MemAvailable|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
T0=$(date +%s)
bash /root/podrun.sh "$N" \
  ARC_KIT=/root/kit_bp \
  ARC_VENV=/root/dsv41/venv_candbp \
  PATH=/root/dsv41/venv_candbp/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_CHALLENGES=/root/data/evalx2t_s2026_challenges.json \
  ARC_SOLUTIONS=/root/data/evalx2t_s2026_solutions.json \
  ARC_TASK_IDS="$IDS" \
  ARC_WORKERS="$W" \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  ARC_MAX_NUM_SEQS=64 \
  ARC_GPU_UTIL=0.92 \
  ARC_CACHE_SALT="$SALT" \
  ARC_STOP_NEW_S=4200 \
  ARC_FORCE_AT_S=4500 \
  ARC_HARD_STOP_S=4620 \
  ARC_KILL_AT_S=4800 \
  ARC_SERVER_EXTRA_ARGS="$EXTRA" >> "$P" 2>&1
echo "$T0" > "$R/arm_t0"
sleep 3
UNTIL=$(( T0 + 5100 ))
python3 -I /root/e057b/scripts/sampler.py --run-dir "$R" --out "$R/metrics10.jsonl" --until "$UNTIL" > "$R/sampler.err" 2>&1 &
S1=$!
( while [ "$(date +%s)" -lt "$UNTIL" ]; do
    m=$(awk '/^(AnonPages|Shmem|MemAvailable|Cached|PageTables|SUnreclaim|KernelStack):/{printf "%s=%.2f ", $1, $2/2^20}' /proc/meminfo | tr -d ':')
    cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null | awk '{printf "%.2f", $1/2^30}')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',')
    p=$(for q in $(pgrep -f "VLLM::|vllm serve|solver_dsv41"); do awk '/^Pss_Anon:/{a=$2} /^Pss_Shmem:/{s=$2} END{printf "%.2f/%.2f;", a/2^20, s/2^20}' /proc/$q/smaps_rollup 2>/dev/null; done)
    sg=$(cat "$R/server.pgid" 2>/dev/null); skb=0
    if [ -n "$sg" ]; then
      skb=$(for q in $(pgrep -g "$sg"); do awk '/^Pss_Anon:/{a=$2} /^Pss_Shmem:/{s=$2} END{print a+s}' /proc/$q/smaps_rollup 2>/dev/null; done | awk '{s+=$1} END{print s+0}')
    fi
    echo "$(date -u +%T) ${m}cgroup_gib=${cg} gpu_mib=${g} server_pss_anon+shmem_gib=$(awk -v k=$skb 'BEGIN{printf "%.2f", k/2^20}') pss_anon/shmem_gib=${p} python_procs=$(pgrep -c python)" >> "$R/memdetail.log"
    if [ "$skb" -gt "$GUARD_KB" ]; then over=$(( ${over:-0} + 1 )); else over=0; fi
    if [ "${over:-0}" -ge 2 ] && [ ! -e "$R/MEMGUARD" ]; then
      sv=$(pgrep -f "kit_bp/solver_dsv41[.]py" | head -1); spg=$(ps -o pgid= -p "$sv" 2>/dev/null | tr -d ' ')
      echo "$(date -u +%T) server PSS $(( skb / 1048576 )) GiB > guard $(( GUARD_KB / 1048576 )) GiB twice: SIGTERM to solver group ${spg:-none}" > "$R/MEMGUARD"
      [ -n "$spg" ] && [ "$spg" != 1 ] && kill -TERM -- "-$spg" 2>/dev/null
    fi
    sleep 30
  done ) &
S2=$!
timeout $(( UNTIL - $(date +%s) )) nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm \
  --format=csv,noheader,nounits -l 30 > "$R/gpu.csv" 2>&1 &
S3=$!
echo "$S1 $S2 $S3" > "$R/samplers.pids"
log "launched; samplers $S1 $S2 $S3 until $(date -u -d @$UNTIL +%T)"
# the server's process group, once it exists (read-only lookup; used only to check it is gone at the end)
SPG=""
while [ $(( $(date +%s) - T0 )) -lt 5000 ]; do
  if [ -z "$SPG" ]; then
    sp=$(pgrep -f "venv_candbp/bin/python[0-9.]* [^ ]*venv_candbp/bin/vllm serve" | head -1)
    [ -n "$sp" ] && SPG=$(ps -o pgid= -p "$sp" | tr -d ' ') && { echo "$SPG" > "$R/server.pgid"; log "server pid $sp pgid $SPG"; }
  fi
  grep -q "^done at +" "$R/run.log" 2>/dev/null && break
  [ -e "$R/MEMGUARD" ] && [ ! -e "$R/.mg_logged" ] && { touch "$R/.mg_logged"; log "MEMORY GUARD: $(cat "$R/MEMGUARD")"; }
  up=$(grep -m1 -o "server up after [0-9]*s" "$R/out/dsv41_logs/solver.log" 2>/dev/null)
  [ -n "$up" ] && [ ! -e "$R/.up_logged" ] && { touch "$R/.up_logged"; log "$up (+$(( $(date +%s) - T0 ))s)"; }
  sleep 20
done
log "run end: $(grep -E '^done at|local score|solver exit code' "$R/run.log" | tr '\n' ' ' | cut -c1-300)"
kill "$S1" "$S2" "$S3" 2>/dev/null
if [ -n "$SPG" ] && pgrep -g "$SPG" > /dev/null; then
  log "server group $SPG still alive after the run: SIGTERM to the group"
  kill -- "-$SPG" 2>/dev/null; sleep 30
  pgrep -g "$SPG" > /dev/null && { kill -9 -- "-$SPG" 2>/dev/null; sleep 5; }
fi
for _ in $(seq 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 500 ] && break; sleep 5
done
log "after: GPU max used ${used} MiB; server group alive: $( [ -n "$SPG" ] && pgrep -g "$SPG" | wc -l || echo n/a); host $(awk '/^(MemAvailable|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
log "ARM DONE $N"
