#!/usr/bin/env bash
# E051 overlay v2 A/B on one 4090 box: arm v2 (byte-level KV parity, then late-phase replays) and arm v1 (late-phase
# replays), each with candidate B's flags (serve_bplus.sh), one after the other. Progress: /root/e051/runs/bplus.progress.
#   setsid nohup bash /root/e051/gpu/bplus_chain.sh < /dev/null > /root/e051/runs/bplus_chain.out 2>&1 &
set -uo pipefail
G=/root/e051/gpu
CONCS=${CONCS:-56 64}
ARM=v2 RUN=bplus_v2 PARITY=1 CONCS="$CONCS" bash "$G/run_bplus.sh"
ARM=v1 RUN=bplus_v1 PARITY=0 CONCS="$CONCS" bash "$G/run_bplus.sh"
echo "$(date -u +%FT%TZ) chain: ALL DONE" >> /root/e051/runs/bplus.progress
