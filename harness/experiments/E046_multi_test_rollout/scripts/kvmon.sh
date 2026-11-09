#!/bin/bash
# E046: poll the E128 server's KV usage, requests, preemptions, prefix-cache and token counters every 30 s into
# <run>/kvmon.log (E042 kvmon.sh, port 18000). Exits once the server session is gone or after 5 failed polls.
R=/root/dsv41/runs/$1; miss=0; SID=$(cat $R/server.pid)
while [ $miss -lt 5 ] && pgrep -s $SID > /dev/null; do
  m=$(curl -s -m 5 localhost:18000/metrics) || { miss=$((miss + 1)); sleep 30; continue; }
  miss=0
  echo "$(date -u +%H:%M:%S) $(echo "$m" | grep -E '^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total|prefix_cache_hits_total|prefix_cache_queries_total|generation_tokens_total|prompt_tokens_total)' | awk '{split($1,a,"{"); printf "%s=%s ", substr(a[1],6), $NF}')" >> $R/kvmon.log
  sleep 30
done
