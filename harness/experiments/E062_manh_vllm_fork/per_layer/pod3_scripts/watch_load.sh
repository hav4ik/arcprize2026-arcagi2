#!/bin/bash
# notebook run: per-rank load timeline from Manh's server.log until the server is up (or 20 min)
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/pod3
read IP PORT < $S/pod3_ssh.txt; last=""
for i in $(seq 1 40); do
  st=$(timeout 30 ssh -o BatchMode=yes -p $PORT root@$IP 'SL=$(ls -d /kaggle/working/dsv41_server_*/server/server.log | tail -1); grep -E "Model loading took|Engram table offloaded|MADV_COLLAPSE|GPU KV cache size|Available KV cache|Graph capturing finished|init engine|Application startup complete|ERROR|Traceback" $SL | sed -E "s/^\((Worker_[A-Z0-9_]+) pid=[0-9]+\) /\1 /" | cut -c1-170; grep -E "MANH SERVER UP|MANH SERVER FAILED" /root/e066/nbrun/run.log | cut -c1-300; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " "' < /dev/null 2>/dev/null)
  [ "$st" != "$last" ] && { echo "== $(date -u +%T)"; echo "$st"; last="$st"; }
  case "$st" in *"MANH SERVER UP"*|*"MANH SERVER FAILED"*) exit 0;; esac
  sleep 30
done
echo "LOAD_WATCH_TIMEOUT $(date -u +%T)"
