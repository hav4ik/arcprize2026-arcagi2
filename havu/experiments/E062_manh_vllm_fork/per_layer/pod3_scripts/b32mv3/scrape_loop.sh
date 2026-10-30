#!/bin/bash
# Scrape the B32 solver's server (port 8100) /metrics every 60 s across server restarts (waits for /v1/models).
while true; do
  until curl -sf -m 5 http://127.0.0.1:8100/v1/models -o /dev/null; do sleep 10; done
  echo "UP $(date +%s)" >> /root/e066/b32mv3/metrics_scrape.txt
  /root/e066/scrape_metrics.sh http://127.0.0.1:8100 /root/e066/b32mv3/metrics_scrape.txt
  echo "DOWN $(date +%s)" >> /root/e066/b32mv3/metrics_scrape.txt
done
