#!/bin/bash
# Wait for v1's NVFP4 probe to finish, stop v1 (by PID), start v2.
until grep -q "nvfp4_probe PROBE:" /root/arc/e037/chain.log; do sleep 60; done
for p in $(pgrep -f "e037_chain[.]sh"); do kill $p; done; sleep 5
for p in $(pgrep -f "vllm.entrypoints.openai.api_server"); do kill $p; done; sleep 25
for p in $(pgrep -f "VLLM::|vllm.v1.engine"); do kill -9 $p; done; sleep 5
echo "[$(date -u +%H:%M)] handoff: v1 stopped after the NVFP4 probe, starting v2" >> /root/arc/e037/chain.log
setsid nohup /root/arc/e037_chain_v2.sh > /root/arc/e037/nohup_v2.log 2>&1 < /dev/null &
