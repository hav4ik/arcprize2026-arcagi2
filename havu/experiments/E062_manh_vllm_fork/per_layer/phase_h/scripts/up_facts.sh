#!/bin/bash
# Load facts for run $1 (after server up or a failure).
RUN=$1; W=/root/kwork_$RUN/dsv41_logs
tail -4 /root/nbruns/$RUN/nb.timeline; head -c 700 $W/supervisor.json 2>/dev/null; echo
grep -a "server up" $W/solver.log | head -3; grep -aE "attempting|deadlines" $W/solver.log | head -2
sed "s/\x1b\[[0-9;]*m//g" $W/server/server.log 2>/dev/null | grep -aE "GPU KV cache size|Maximum concurrency|Capped the PyTorch allocator|Model loading took|CUDA out of memory|Traceback|Error|EngineDead|bidirectional|non-default args" | cut -c1-1500 | head -24
cat /root/nbruns/$RUN/uvcache_cleanup.log 2>/dev/null; tail -3 /root/nbruns/$RUN/samp.alerts 2>/dev/null | cut -c1-300; df -h / | tail -1
