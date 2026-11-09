#!/usr/bin/env bash
# E051 B+ re-plan (2026-10-27 08:50 UTC). v2 at 56 late-phase streams collapsed to ~1% host hits: 16 GiB = 37.7k host
# blocks (4.8M MLA tokens) < 56 x 82.5k-token p50 prompts, so the first queue cycle overflows the host and LRU evicts
# each waiting stream's prefix before it is admitted. The v1/v2 A/B therefore runs below that cliff (48 streams, where
# v1 got 948 tok/s on box 3), plus v1 at 56 as the control that the collapse is capacity, not v2.
#   setsid nohup bash /root/e051/gpu/bplus_chain2.sh < /dev/null > /root/e051/runs/bplus_chain2.out 2>&1 &
set -uo pipefail
G=/root/e051/gpu
ARM=v2 RUN=bplus_v2b PARITY=0 CONCS="48" bash "$G/run_bplus.sh"
ARM=v1 RUN=bplus_v1 PARITY=0 CONCS="48 56" bash "$G/run_bplus.sh"
echo "$(date -u +%FT%TZ) chain2: ALL DONE" >> /root/e051/runs/bplus.progress
