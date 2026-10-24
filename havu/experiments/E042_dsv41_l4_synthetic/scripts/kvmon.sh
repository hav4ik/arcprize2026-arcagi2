#!/bin/bash
# E042: poll the server's KV usage, running/waiting requests and preemptions every 30 s into <run>/kvmon.log.
# Exits once the run's server session is gone (else it would poll the next run's server on the same port), or after
# the server has been unreachable for 5 polls.
R=/root/dsv41/runs/$1; miss=0; SID=$(cat $R/server.pid)
while [ $miss -lt 5 ] && pgrep -s $SID > /dev/null; do
  m=$(curl -s -m 5 localhost:8100/metrics) || { miss=$((miss + 1)); sleep 30; continue; }
  miss=0
  echo "$(date -u +%H:%M:%S) $(echo "$m" | grep -E '^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total|prefix_cache_hits_total|prefix_cache_queries_total|generation_tokens_total)' | awk '{split($1,a,"{"); printf "%s=%s ", substr(a[1],6), $NF}')" >> $R/kvmon.log
  sleep 30
done
