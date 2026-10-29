#!/bin/bash
# at a UTC time: per-rank load state + compaction counters (for the 8-min flag)
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/pod3
T=$1; while [ "$(date -u +%H%M%S)" \< "$T" ]; do sleep 15; done
read IP PORT < $S/pod3_ssh.txt
timeout 60 ssh -o BatchMode=yes -p $PORT root@$IP 'date -u +%T; SL=$(ls -d /kaggle/working/dsv41_server_*/server/server.log | tail -1); grep -E "Model loading took|MADV_COLLAPSE|GPU KV cache size|Graph capturing finished|Application startup complete" $SL | sed -E "s/\x1b\[[0-9;]*m//g" | cut -c1-160; grep -E "MANH SERVER UP|MANH SERVER FAILED" /root/e066/nbrun/run.log | cut -c1-200; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " "; echo; grep -E "^compact_(stall|fail|success|migrate_scanned) " /proc/vmstat | tr "\n" " "; echo; ps -eo pid,pcpu,rss,args | grep "[V]LLM::Worker" | cut -c1-80' < /dev/null
