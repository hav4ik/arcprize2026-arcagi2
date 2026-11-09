#!/bin/bash
# Wait for the server (any 200 from /v1/models), then scrape /metrics every 60 s. Read-only.
until curl -sf -m 5 http://127.0.0.1:18084/v1/models -o /dev/null; do sleep 10; done
exec /root/e066/scrape_metrics.sh http://127.0.0.1:18084 /root/e066/nbrun/metrics_scrape.txt
