#!/usr/bin/env bash
# E059 B16 on Vast 4x4090 box 57983124: restart-under-load test of candidate "B16" (the Kaggle-smoked notebook's config:
# candidate B = venv_candb + kit_b, 56 workers, gate 0.85 + queue hold, 16 GiB KV offload), the 4090 twin of the L4 run
# experiments/E059_restart_under_load/launch_b32kill.sh (32 GiB). The real solver through /root/podrun.sh +
# /root/local_run.py (as E057b run_arm.sh), on the 90 heaviest evalx2t_s2026 tasks, 3 h deadlines
# (stop-new 9600 / force 10200 / hard stop 10500 / kill switch 10800 s), KV pool pinned to Kaggle's
# (--num-gpu-blocks-override 21490 at util 0.92).
# Engine kill: killer.sh <run_dir> 5400 SIGKILLs this run's own VLLM::Worker_TP1 (found among the solver's descendants)
# 1.5 h after "server up" and samples host memory every 2 s from 5 min before to 20 min after (mem2s.log).
# Samplers: sampler.py (10 s: /metrics, status.json, host memory), memdetail (30 s: meminfo, cgroup anon/shmem, GPU,
# per-process PSS, cpu.stat throttling), mem2s_guard.log (2 s: host AnonPages/Shmem and the cgroup anon/shmem/file),
# nvidia-smi (30 s).
# Memory guard (protects the box): every 2 s; if host AnonPages + Shmem exceeds 85% of the cgroup limit in 2 samples in a
# row, the solver's process group gets SIGTERM (it writes its submission and stops its server).
# Every server process group seen is recorded (server_pgids.txt); at the end any that is still alive is stopped by
# PGID (never by name), and the GPUs must be idle.
#
#   setsid -f bash /root/e059/scripts/run_b16.sh [name] [kill after s] > /root/e059/runs/run_b16.driver.out 2>&1 < /dev/null
#   (progress: /root/e059/runs/<name>.progress)
set -uo pipefail
N=${1:-e059_b16kill}; KILL_AFTER=${2:-5400}
OFF=17179869184
R=/root/runs/$N
P=/root/e059/runs/$N.progress
mkdir -p /root/e059/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$P"; }
[ ! -e "$R" ] || { log "$R exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { log "a GPU holds ${used} MiB: not starting"; exit 1; }
if pgrep -f 'python[0-9.]* -u [^ ]*/[l]ocal_run[.]py|python[0-9.]* -u [^ ]*/[s]olver_dsv41[.]py' > /dev/null; then
  log "a solver is already running: not starting"; exit 1; fi
IDS=$(tr -d ' \n' < /root/e059/workloads/evalx2t_s2026_heavy90_ids.txt)
EXTRA='["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":'"$OFF"',\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}", "--num-gpu-blocks-override", "21490"]'
SALT=${N}_$(date -u +%H%M%S)
# The image's environment carries credentials and its own Qwen vLLM settings; none of them belong in the solver,
# the server or the tool sandbox (Kaggle has none of them).
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HUGGINGFACE_HUB_TOKEN$|HF_API_TOKEN$|WANDB_API_KEY$|CONTAINER_API_KEY$|JUPYTER_TOKEN$|OPEN_BUTTON_TOKEN$)'); do
  unset "$v"; done
MT_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
CG_B=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
[[ "$CG_B" =~ ^[0-9]+$ ]] || CG_B=$(( MT_KB * 1024 ))
GUARD_KB=$(( CG_B / 1024 * 85 / 100 ))
log "start $N: workers 56, offload $OFF B, kill TP1 at server up + ${KILL_AFTER}s, MemTotal $(( MT_KB / 1048576 )) GiB, cgroup max $CG_B B ($(( CG_B / 1073741824 )) GiB), guard host AnonPages+Shmem > $(( GUARD_KB / 1048576 )) GiB (85%), cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null), salt $SALT, $(echo "$IDS" | tr ',' '\n' | wc -l) task ids; host $(awk '/^(MemAvailable|AnonPages|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
T0=$(date +%s)
bash /root/podrun.sh "$N" \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/dsv41/venv_candb \
  PATH=/root/dsv41/venv_candb/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_CHALLENGES=/root/data/evalx2t_s2026_challenges.json \
  ARC_SOLUTIONS=/root/data/evalx2t_s2026_solutions.json \
  ARC_TASK_IDS="$IDS" \
  ARC_WORKERS=56 \
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
  ARC_STOP_NEW_S=9600 \
  ARC_FORCE_AT_S=10200 \
  ARC_HARD_STOP_S=10500 \
  ARC_KILL_AT_S=10800 \
  ARC_SERVER_EXTRA_ARGS="$EXTRA" >> "$P" 2>&1
echo "$T0" > "$R/arm_t0"
# the solver's pid for killer.sh (its own fallback lookup expects "python solver_dsv41.py", but the notebook runs
# "python -u solver_dsv41.py")
SV=""
for _ in $(seq 120); do
  SV=$(pgrep -f '^/root/dsv41/venv_candb/bin/python[0-9.]* -u /root/kit_b/solver_dsv41[.]py$' | head -1)
  [ -n "$SV" ] && break; sleep 1
done
[ -n "$SV" ] || { log "solver process not found 120 s after launch"; }
echo "$SV" > "$R/solver.pid"
log "solver pid ${SV:-none} pgid $(ps -o pgid= -p "${SV:-1}" 2>/dev/null | tr -d ' ')"
setsid nohup bash /root/e059/scripts/killer.sh "$R" "$KILL_AFTER" > "$R/killer.out" 2>&1 < /dev/null &
log "killer.sh started (pid $!): kill TP1 at server up + ${KILL_AFTER}s"
sleep 2
UNTIL=$(( T0 + 11400 ))
python3 -I /root/e059/scripts/sampler.py --run-dir "$R" --out "$R/metrics10.jsonl" --until "$UNTIL" > "$R/sampler.err" 2>&1 &
S1=$!
( i=0; over=0
  while [ "$(date +%s)" -lt "$UNTIL" ]; do
    read -r an sh <<< "$(awk '/^AnonPages:/{a=$2} /^Shmem:/{s=$2} END{print a, s}' /proc/meminfo)"
    hkb=$(( an + sh ))
    # every 2 s: host-wide AnonPages/Shmem (includes other tenants of the host) and this container's cgroup anon/shmem
    echo "$(date -u +%T) host_anon=$(( an / 1024 )) host_shmem=$(( sh / 1024 )) $(awk '/^(anon|shmem|file) /{printf "cg_%s=%d ", $1, $2/1048576}' /sys/fs/cgroup/memory.stat 2>/dev/null)cg_current=$(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1048576 )) vllm_procs=$(pgrep -c -f 'VLLM::|vllm serve') (MiB)" >> "$R/mem2s_guard.log"
    if [ "$hkb" -gt "$GUARD_KB" ]; then over=$(( over + 1 )); else over=0; fi
    if [ "$over" -ge 2 ] && [ ! -e "$R/MEMGUARD" ]; then
      sv=$(cat "$R/solver.pid" 2>/dev/null); spg=$(ps -o pgid= -p "${sv:-1}" 2>/dev/null | tr -d ' ')
      echo "$(date -u +%T) host AnonPages+Shmem $(( hkb / 1048576 )) GiB > guard $(( GUARD_KB / 1048576 )) GiB twice: SIGTERM to solver group ${spg:-none}" > "$R/MEMGUARD"
      [ -n "$spg" ] && [ "$spg" != 1 ] && [ "$spg" = "$sv" ] && kill -TERM -- "-$spg" 2>/dev/null
    fi
    if [ $(( i % 15 )) -eq 0 ]; then
      m=$(awk '/^(AnonPages|Shmem|MemAvailable|Cached|Mlocked|PageTables|SUnreclaim):/{printf "%s=%.2f ", $1, $2/2^20}' /proc/meminfo | tr -d ':')
      cg=$(awk '/^(anon|shmem|file) /{printf "cg_%s=%.2f ", $1, $2/2^30}' /sys/fs/cgroup/memory.stat 2>/dev/null)
      cgc=$(awk '{printf "%.2f", $1/2^30}' /sys/fs/cgroup/memory.current 2>/dev/null)
      thr=$(awk '/^(nr_throttled|throttled_usec|usage_usec) /{printf "%s=%s ", $1, $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null)
      g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',')
      pss=$(for q in $(pgrep -f "VLLM::|vllm serve"); do awk '/^Pss_Anon:/{a=$2} /^Pss_Shmem:/{s=$2} END{printf "%.2f/%.2f;", a/2^20, s/2^20}' /proc/$q/smaps_rollup 2>/dev/null; done)
      echo "$(date -u +%T) ${m}${cg}cgroup_gib=${cgc} ${thr}gpu_mib=${g} vllm_pss_anon/shmem_gib=${pss} vllm_procs=$(pgrep -c -f 'VLLM::|vllm serve') python_procs=$(pgrep -c python)" >> "$R/memdetail.log"
    fi
    i=$(( i + 1 )); sleep 2
  done ) &
S2=$!
timeout $(( UNTIL - $(date +%s) )) nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm \
  --format=csv,noheader,nounits -l 30 > "$R/gpu.csv" 2>&1 &
S3=$!
echo "$S1 $S2 $S3" > "$R/samplers.pids"
log "launched; samplers $S1 $S2 $S3 until $(date -u -d @$UNTIL +%T)"
# every server process group seen (read-only lookup; used to check they are all gone at the end)
touch "$R/server_pgids.txt"
while [ $(( $(date +%s) - T0 )) -lt 11200 ]; do
  for sp in $(pgrep -f "venv_candb/bin/python[0-9.]* [^ ]*venv_candb/bin/vllm serve"); do
    pg=$(ps -o pgid= -p "$sp" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && ! grep -q "^$pg " "$R/server_pgids.txt" && { echo "$pg $sp $(date -u +%T)" >> "$R/server_pgids.txt"; log "server pid $sp pgid $pg"; }
  done
  grep -q "^done at +" "$R/run.log" 2>/dev/null && break
  [ -e "$R/MEMGUARD" ] && [ ! -e "$R/.mg_logged" ] && { touch "$R/.mg_logged"; log "MEMORY GUARD: $(cat "$R/MEMGUARD")"; }
  up=$(grep -c "server up after" "$R/out/dsv41_logs/solver.log" 2>/dev/null)
  if [ "${up:-0}" -gt "$(cat "$R/.up_n" 2>/dev/null || echo 0)" ]; then
    echo "$up" > "$R/.up_n"; log "$(grep "server up after" "$R/out/dsv41_logs/solver.log" | tail -1 | cut -c1-200) (+$(( $(date +%s) - T0 ))s)"; fi
  [ -s "$R/killer.log" ] && [ "$(wc -l < "$R/killer.log")" -gt "$(cat "$R/.kl_n" 2>/dev/null || echo 0)" ] && {
    log "killer: $(tail -1 "$R/killer.log")"; wc -l < "$R/killer.log" > "$R/.kl_n"; }
  sleep 20
done
log "run end: $(grep -E '^done at|local score|solver exit code' "$R/run.log" | tr '\n' ' ' | cut -c1-300)"
sleep 30
kill "$S1" "$S2" "$S3" 2>/dev/null
while read -r pg sp ts; do
  [ -n "$pg" ] || continue
  if pgrep -g "$pg" > /dev/null; then
    if ps -o args= -g "$pg" | grep -qE "venv_candb|^VLLM::"; then
      log "server group $pg (seen $ts) still alive after the run: SIGTERM to the group"
      kill -- "-$pg" 2>/dev/null; sleep 30
      pgrep -g "$pg" > /dev/null && { kill -9 -- "-$pg" 2>/dev/null; sleep 5; }
    else
      log "pgid $pg is alive but is not a server of ours: left alone"
    fi
  fi
done < "$R/server_pgids.txt"
for _ in $(seq 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 500 ] && break; sleep 5
done
alive=0; while read -r pg sp ts; do [ -n "$pg" ] && pgrep -g "$pg" > /dev/null && alive=$(( alive + 1 )); done < "$R/server_pgids.txt"
log "after: GPU max used ${used} MiB; server groups seen $(wc -l < "$R/server_pgids.txt"), alive $alive; VLLM procs $(pgrep -c -f 'VLLM::'); host $(awk '/^(MemAvailable|AnonPages|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
log "RUN DONE $N"
