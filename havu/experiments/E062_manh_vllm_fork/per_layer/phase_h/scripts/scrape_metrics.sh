#!/bin/bash
# Scrape the server's /metrics every 60 s (selected counters/gauges) until it stops answering. Usage: scrape_metrics.sh <base> <out>
B=$1; O=$2
while curl -sf -m 10 "$B/metrics" -o /tmp/scrape_m.txt; do
  { echo "T $(date +%s)"; grep -E "^vllm:(prefix_cache_(queries|hits)_total|external_prefix_cache_(queries|hits)_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total|kv_cache_usage_perc|num_requests_(running|waiting)|simple_kv_offload_(used|pending_store|pending_load)[a-z_]*|mm_cache_(queries|hits)_total|kv_offload[a-z_]*)" /tmp/scrape_m.txt | sed 's/{[^}]*}//'; } >> "$O"
  sleep 60
done
