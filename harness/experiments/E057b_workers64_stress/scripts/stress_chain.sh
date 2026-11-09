#!/usr/bin/env bash
# E057b: the two arms back to back on the same workload, one server at a time: A = 56 workers (the smoke-tested
# Kaggle config) first, then B = 64 workers, both at 16 GiB offload (coordinator update, 2026-10-27).
#   setsid -f bash /root/e057b/scripts/stress_chain.sh > /root/e057b/runs/stress_chain.driver.out 2>&1 < /dev/null
set -uo pipefail
L=/root/e057b/runs/stress_chain.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
log "STRESS START; host $(awk '/^(MemTotal|MemAvailable|Shmem):/{printf "%s %.1f GiB ", $1, $2/1048576}' /proc/meminfo)"
for arm in "e057b_w56 56" "e057b_w64 64"; do
  set -- $arm
  log "arm $1 ($2 workers) start"
  bash /root/e057b/scripts/run_arm.sh "$1" "$2"
  rc=$?
  log "arm $1 end rc=$rc: $(tail -2 /root/e057b/runs/$1.progress | tr '\n' ' ' | cut -c1-300)"
  grep -q "ARM DONE $1" /root/e057b/runs/$1.progress || { log "STRESS FAILED at $1"; log "STRESS END"; exit 1; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 500 ] || { log "GPUs not idle after $1 (${used} MiB): stopping the chain"; log "STRESS END"; exit 1; }
  sleep 20
done
log "STRESS OK"
log "STRESS END"
