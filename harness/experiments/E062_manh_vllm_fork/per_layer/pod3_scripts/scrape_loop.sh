#!/bin/bash
# Scrape /metrics every 60 s across server restarts: wait for /v1/models, scrape until it stops answering, repeat.
while true; do
  until curl -sf -m 5 http://127.0.0.1:18084/v1/models -o /dev/null; do sleep 10; done
  echo "UP $(date +%s)" >> /root/e066/r3/metrics_scrape.txt
  /root/e066/scrape_metrics.sh http://127.0.0.1:18084 /root/e066/r3/metrics_scrape.txt
  echo "DOWN $(date +%s)" >> /root/e066/r3/metrics_scrape.txt
done
