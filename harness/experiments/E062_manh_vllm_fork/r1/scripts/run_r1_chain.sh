#!/usr/bin/env bash
# E062 R1 chain: B32 default padding, then 64,64 (no padding for E128's TP4 shards: N 1152, K 576); if 64,64 does not
# start, 256,64 (pad N only) and 64,128 (pad K only). One arm at a time; each arm stops its own server group.
#   setsid -f bash /root/e062/r1/run_r1_chain.sh > /root/e062/runs/chain.out 2>&1 < /dev/null
X=/root/e062/r1
RUN=r1_def PAD= bash $X/run_r1_arm.sh
RUN=r1_p6464 PAD=64,64 bash $X/run_r1_arm.sh && { echo "$(date -u +%FT%TZ) CHAIN DONE" >> /root/e062/runs/chain.progress; exit 0; }
RUN=r1_p25664 PAD=256,64 bash $X/run_r1_arm.sh
RUN=r1_p64128 PAD=64,128 bash $X/run_r1_arm.sh
echo "$(date -u +%FT%TZ) CHAIN DONE" >> /root/e062/runs/chain.progress
