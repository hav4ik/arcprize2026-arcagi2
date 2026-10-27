#!/usr/bin/env bash
# E059 hang rehearsal on Vast 4x4090 box 57983124: candidate B32.1 (B32 + S20 progress watchdog, S21 re-launch hold,
# S22 post-restart ramp) under the Kaggle smoke's hang test, before any Kaggle quota is spent. The real solver
# (kit_b1 = kit_b with the B32.1 solver dee879ce) through /root/podrun.sh + /root/local_run.py, the notebook's smoke
# settings: the 24 smoke-stress tasks, GPU KV pool cut to 4500 blocks, 34-min smoke deadlines (stop-new 1740 / force 1860
# / hard stop 1920 / kill switch 2010 s), watchdog 60 s, restart allowed until 300 s before the hard stop, ramp 16,8,
# 32 GiB KV offload. freezer.sh SIGSTOPs this run's own VLLM::EngineCore 20 s after "server up" (as cell 2c does on
# Kaggle) and resumes it after 420 s if no restart happened. Samplers and the memory guard as run_b16.sh.
#
#   setsid -f bash /root/e059/scripts/run_hang.sh [name] > /root/e059/runs/run_hang.driver.out 2>&1 < /dev/null
set -uo pipefail
N=${1:-e059_hang}
OFF=34359738368
R=/root/runs/$N
P=/root/e059/runs/$N.progress
mkdir -p /root/e059/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$P"; }
[ ! -e "$R" ] || { log "$R exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { log "a GPU holds ${used} MiB: not starting"; exit 1; }
if pgrep -f 'python[0-9.]* -u [^ ]*/[l]ocal_run[.]py|python[0-9.]* -u [^ ]*/[s]olver_dsv41[.]py' > /dev/null; then
  log "a solver is already running: not starting"; exit 1; fi
IDS=25ff71a9,27a28665,00576224,337b420f,25c199f5,12997ef3,2ccd9fef,0ca9ddb6,412b6263,351d6448,178fcbfb,137f0df0,2204b7a8,22eb0ac0,292dd178,05f2a901,2c737e39,230f2e48,20fb2937,3bdb4ada,2753e76c,20818e16,41e4d17e,13f06aa5
EXTRA='["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":'"$OFF"',\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}", "--num-gpu-blocks-override", "4500"]'
SALT=${N}_$(date -u +%H%M%S)
# The image's environment carries credentials and its own Qwen vLLM settings; none of them belong in the solver,
# the server or the tool sandbox (Kaggle has none of them).
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HUGGINGFACE_HUB_TOKEN$|HF_API_TOKEN$|WANDB_API_KEY$|CONTAINER_API_KEY$|JUPYTER_TOKEN$|OPEN_BUTTON_TOKEN$)'); do
  unset "$v"; done
MT_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
CG_B=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
[[ "$CG_B" =~ ^[0-9]+$ ]] || CG_B=$(( MT_KB * 1024 ))
GUARD_KB=$(( CG_B / 1024 * 85 / 100 ))
log "start $N: workers 56, offload $OFF B, freeze EngineCore at server up + 20s, MemTotal $(( MT_KB / 1048576 )) GiB, cgroup max $CG_B B ($(( CG_B / 1073741824 )) GiB), guard host AnonPages+Shmem > $(( GUARD_KB / 1048576 )) GiB (85%), cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null), salt $SALT, $(echo "$IDS" | tr ',' '\n' | wc -l) task ids; host $(awk '/^(MemAvailable|AnonPages|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
T0=$(date +%s)
bash /root/podrun.sh "$N" \
  ARC_KIT=/root/kit_b1 \
  ARC_VENV=/root/dsv41/venv_candb \
  PATH=/root/dsv41/venv_candb/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_CHALLENGES=/root/data/arc-agi_training_challenges.json \
  ARC_SOLUTIONS=/root/data/arc-agi_training_solutions.json \
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
  ARC_STOP_NEW_S=1740 \
  ARC_FORCE_AT_S=1860 \
  ARC_HARD_STOP_S=1920 \
  ARC_KILL_AT_S=2010 \
  ARC_STALL_RESTART_S=60 \
  ARC_CONN_HOLD=1 \
  ARC_RESTART_RAMP=16,8 \
  ARC_RESTART_MIN_LEFT_S=300 \
  ARC_SERVER_EXTRA_ARGS="$EXTRA" >> "$P" 2>&1
echo "$T0" > "$R/arm_t0"
# the solver's pid for freezer.sh (its own fallback lookup expects "python solver_dsv41.py", but the notebook runs
# "python -u solver_dsv41.py")
SV=""
for _ in $(seq 120); do
  SV=$(pgrep -f '^/root/dsv41/venv_candb/bin/python[0-9.]* -u /root/kit_b1/solver_dsv41[.]py$' | head -1)
  [ -n "$SV" ] && break; sleep 1
done
[ -n "$SV" ] || { log "solver process not found 120 s after launch"; }
echo "$SV" > "$R/solver.pid"
log "solver pid ${SV:-none} pgid $(ps -o pgid= -p "${SV:-1}" 2>/dev/null | tr -d ' ')"
setsid nohup bash /root/e059/scripts/freezer.sh "$R" 20 420 > "$R/freezer.out" 2>&1 < /dev/null &
log "freezer.sh started (pid $!): SIGSTOP EngineCore at server up + 20s"
sleep 2
UNTIL=$(( T0 + 2400 ))
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
while [ $(( $(date +%s) - T0 )) -lt 2300 ]; do
  for sp in $(pgrep -f "venv_candb/bin/python[0-9.]* [^ ]*venv_candb/bin/vllm serve"); do
    pg=$(ps -o pgid= -p "$sp" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && ! grep -q "^$pg " "$R/server_pgids.txt" && { echo "$pg $sp $(date -u +%T)" >> "$R/server_pgids.txt"; log "server pid $sp pgid $pg"; }
  done
  grep -q "^done at +" "$R/run.log" 2>/dev/null && break
  [ -e "$R/MEMGUARD" ] && [ ! -e "$R/.mg_logged" ] && { touch "$R/.mg_logged"; log "MEMORY GUARD: $(cat "$R/MEMGUARD")"; }
  up=$(grep -c "server up after" "$R/out/dsv41_logs/solver.log" 2>/dev/null)
  if [ "${up:-0}" -gt "$(cat "$R/.up_n" 2>/dev/null || echo 0)" ]; then
    echo "$up" > "$R/.up_n"; log "$(grep "server up after" "$R/out/dsv41_logs/solver.log" | tail -1 | cut -c1-200) (+$(( $(date +%s) - T0 ))s)"; fi
  [ -s "$R/freezer.log" ] && [ "$(wc -l < "$R/freezer.log")" -gt "$(cat "$R/.fl_n" 2>/dev/null || echo 0)" ] && {
    log "freezer: $(tail -1 "$R/freezer.log")"; wc -l < "$R/freezer.log" > "$R/.fl_n"; }
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
