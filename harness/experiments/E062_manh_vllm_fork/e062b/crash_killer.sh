#!/bin/bash
# E062b crash test (on pod #1): wait until the e062b_crash server has been up >= LOAD_S seconds and >= MINRUN requests
# are running, then SIGKILL the solver's vLLM EngineCore (by PID: the VLLM::EngineCore process under this run's
# solver), and log what happened. Never a name pattern for the kill.
R=/root/runs/e062b_crash; L=$R/killer.log; LOAD_S=${LOAD_S:-360}; MINRUN=${MINRUN:-75}
log() { echo "$(date -u +%T) $*" >> $L; }
SL=$R/out/dsv41_logs/solver.log
until grep -q "server up after" $SL 2>/dev/null; do sleep 5; done
UP=$(date +%s); log "server up seen: $(grep -m1 'server up after' $SL | cut -c1-120)"
while :; do
  run=$(grep -o "Running: [0-9]* reqs" $R/out/dsv41_logs/server.log 2>/dev/null | tail -1 | grep -o "[0-9]*")
  [ $(( $(date +%s) - UP )) -ge $LOAD_S ] && [ "${run:-0}" -ge $MINRUN ] && break
  [ $(( $(date +%s) - UP )) -ge $(( LOAD_S + 300 )) ] && { log "running only ${run:-0} after $(( LOAD_S + 300 )) s; killing anyway"; break; }
  sleep 5
done
SOLVER=$(ps -eo pid,args | awk '$4 == "/root/kit_b/solver_dsv41.py" {print $1}' | head -1)
# EngineCore under this solver: walk each VLLM::EngineCore's ancestors up to the solver PID
EC=""
for p in $(ps -eo pid,args | awk '$2 == "VLLM::EngineCore" {print $1}'); do
  q=$p; while [ "$q" -gt 1 ]; do q=$(ps -o ppid= -p $q | tr -d " "); [ "$q" = "$SOLVER" ] && { EC=$p; break; }; [ -z "$q" ] && break; done
  [ -n "$EC" ] && break
done
[ -n "$EC" ] || { log "no EngineCore under solver $SOLVER; not killing"; exit 1; }
log "killing EngineCore $EC (solver $SOLVER), running ${run:-?}; gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
kill -9 $EC; KT=$(date +%s); log "SIGKILL sent"
for i in $(seq 120); do
  sleep 5
  [ $(( i % 6 )) -eq 0 ] && log "t+$(( $(date +%s) - KT ))s gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ') procs $(ps -eo stat,args | awk '$2 ~ /vllm|VLLM/' | wc -l) zombies $(ps -eo stat | grep -c '^Z')"
done
log "killer done"
