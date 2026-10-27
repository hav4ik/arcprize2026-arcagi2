#!/usr/bin/env bash
# E056: start the offload-only arm once the candB arm has stopped its server.
# Launch detached (a plain `setsid nohup ... &` inside `ssh '...'` kept the ssh channel open on this box):
#   ssh box 'setsid -f bash /root/e056/gpu/chain_offload.sh > /root/e056/runs/late_offload.driver.out 2>&1 < /dev/null'
until grep -q "late_candb: stopped" /root/e056/runs/late.progress; do sleep 15; done
sleep 20
cd /root/e056 && ARM=offload RUN=late_offload exec bash gpu/run_arm.sh
