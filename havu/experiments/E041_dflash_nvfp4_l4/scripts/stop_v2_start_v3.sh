#!/bin/bash
for p in $(pgrep -f "e037_chain_v[2].sh"); do kill $p; done
/root/arc/list_solvers.sh 1 > /dev/null
echo "[$(date -u +%H:%M)] v2 stopped (skipping the NVFP4+DFlash probe); starting v3" >> /root/arc/e037/chain.log
setsid nohup /root/arc/e037_chain_v3.sh > /root/arc/e037/nohup_v3.log 2>&1 < /dev/null &
sleep 3; tail -2 /root/arc/e037/chain.log
