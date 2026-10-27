#!/usr/bin/env bash
# E051 B+ extra arm (dsv4-infer, 2026-10-27): v2 at 64 late-phase streams with 32 GiB private host memory
# (cpu_bytes_to_use 34359738368 = 8 GiB per rank, 75.4k host blocks, ~9.6M MLA tokens), B+'s likely config if 64
# workers survive. Starts after bplus_chain2.sh's process group (pgid argument) has exited and the GPUs are free.
#   setsid nohup bash /root/e051/gpu/bplus_chain3.sh <chain2 pgid> < /dev/null > /root/e051/runs/bplus_chain3.out 2>&1 &
set -uo pipefail
G=/root/e051/gpu
PROG=/root/e051/runs/bplus.progress
WAIT_PG=${1:?usage: bplus_chain3.sh CHAIN2_PGID}
log() { echo "$(date -u +%FT%TZ) $*" >> "$PROG"; }
log "chain3: queued behind pgid $WAIT_PG"
while pgrep -g "$WAIT_PG" > /dev/null; do sleep 30; done
sleep 20
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${used:-99999}" -lt 500 ] || { log "chain3: GPUs busy (max ${used} MiB used), not starting"; exit 1; }
ARM=v2 RUN=bplus_v2x32 PARITY=0 CONCS="64" OFFLOAD_BYTES=34359738368 bash "$G/run_bplus.sh"
log "chain3: ALL DONE"
