#!/bin/bash
# Read-only: scrape the candidate server's /metrics (port 18084) every 60 s, across restarts, until the notebook ends.
# Format (as pod 3's): "UP <epoch>", then per scrape "T <epoch>" + selected vllm:* lines (labels stripped), "DOWN <epoch>".
# Usage: scrape_loop.sh <out> <nb.timeline>
O=$1; TL=$2; B=http://127.0.0.1:18084; TMP=$(mktemp)
done_() { grep -q NOTEBOOK_DONE "$TL" 2>/dev/null; }
while ! done_; do
  until curl -sf -m 5 $B/v1/models -o /dev/null || done_; do sleep 10; done
  done_ && break
  echo "UP $(date +%s)" >> "$O"
  while curl -sf -m 10 "$B/metrics" -o $TMP; do
    { echo "T $(date +%s)"; grep -E "^vllm:(prefix_cache_(queries|hits)_total|external_prefix_cache_(queries|hits)_total|prompt_tokens_total|generation_tokens_total|num_preemptions_total|kv_cache_usage_perc|num_requests_(running|waiting)|kv_offload[a-z_]*)" $TMP | sed 's/{[^}]*}//'; } >> "$O"
    sleep 60
  done
  echo "DOWN $(date +%s)" >> "$O"
done
rm -f $TMP
