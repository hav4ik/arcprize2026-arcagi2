#!/bin/bash
# E059: kill one TP worker of OUR vLLM server mid-run (simulated engine death) and sample host memory every 2 s around
# it, to check the solver's restart path with KV offload on: old server memory (Engram shm + pinned pool) released
# before the new server allocates its own, jobs re-queued (S15), throughput recovered.
#   killer.sh <run_dir> <seconds after "server up" to kill>
# Never kills by name: the worker is found among the descendants of this run's own solver process.
RUN=$1; AFTER=$2; LOG=$RUN/killer.log; MEM=$RUN/mem2s.log
log() { echo "$(date -u +%T) $*" >> $LOG; }
until grep -q "server up" $RUN/out/dsv41_logs/solver.log 2>/dev/null; do sleep 10; done
log "server up seen; killing in $AFTER s"
sleep $(( AFTER - 300 ))
( end=$(( $(date +%s) + 1500 )); while [ $(date +%s) -lt $end ]; do
    m=$(awk '/^(AnonPages|Shmem|MemAvailable):/{printf "%s=%.2f ", $1, $2/2^20}' /proc/meminfo | tr -d ':')
    n=$(ps -eo args | grep -c "^VLLM::")
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',')
    echo "$(date -u +%T) ${m}vllm_procs=$n gpu_mib=$g" >> $MEM; sleep 2; done ) &
sleep 300
SOLVER=$(cat $RUN/solver.pid 2>/dev/null)
[ -z "$SOLVER" ] && SOLVER=$(ps -eo pid,args | awk '$2 ~ /python/ && $NF ~ /solver_dsv41.py$/ {print $1}' | head -1)  # last field: the notebook runs python -u <path>
desc() { local p=$1; for c in $(ps -o pid= --ppid $p); do echo $c; desc $c; done; }
W=""; for p in $(desc $SOLVER); do a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); [[ "$a" == VLLM::Worker_TP1* ]] && W=$p; done
if [ -z "$W" ]; then log "no TP1 worker found under solver $SOLVER; abort"; exit 1; fi
log "solver $SOLVER; SIGKILL worker TP1 pid $W"
kill -KILL $W
wait
log "memory sampling done"
