#!/bin/bash
# E059 hang rehearsal: SIGSTOP this run's own VLLM::EngineCore (found among the solver's descendants, never by name alone)
# AFTER seconds after "server up"; resume it after MAX seconds if the solver has not restarted the server.
#   freezer.sh <run_dir> <after_s> <max_s>
RUN=$1; AFTER=$2; MAX=$3; LOG=$RUN/freezer.log
log() { echo "$(date -u +%T) $*" >> $LOG; }
until grep -q "server up" $RUN/out/dsv41_logs/solver.log 2>/dev/null; do sleep 5; done
log "server up seen; freezing the engine in $AFTER s"; sleep $AFTER
SOLVER=$(cat $RUN/solver.pid 2>/dev/null)
[ -z "$SOLVER" ] && SOLVER=$(ps -eo pid,args | awk '$2 ~ /python/ && $NF ~ /solver_dsv41.py$/ {print $1}' | head -1)
desc() { local p=$1; for c in $(ps -o pid= --ppid $p); do echo $c; desc $c; done; }
E=""; for p in $(desc $SOLVER); do a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); [[ "$a" == VLLM::EngineCore* ]] && E=$p; done
[ -z "$E" ] && { log "no EngineCore under solver $SOLVER"; exit 1; }
kill -STOP $E; log "SIGSTOP EngineCore $E (solver $SOLVER)"
for i in $(seq $((MAX / 5))); do grep -q "server start (restart" $RUN/out/dsv41_logs/solver.log && { log "restart seen after $((i*5)) s"; exit 0; }; sleep 5; done
kill -CONT $E 2>/dev/null; log "no restart after $MAX s: resumed the engine"
