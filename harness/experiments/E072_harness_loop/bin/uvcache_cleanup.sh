#!/bin/bash
# pod-only accommodation as in V12 replicas: delete the ~20 GB uv cache once the install cells are done (cell 11).
R=$1; t0=$(date +%s)
until grep -q "END   cell 11:" $R/nb.timeline 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && exit 1
  grep -q NOTEBOOK_DONE $R/nb.timeline 2>/dev/null && exit 1
  sleep 5
done
rm -rf /tmp/dsv41/cache/uv; echo "$(date -u +%T) uv cache deleted; free $(df -h / | tail -1 | awk '{print $4}')"
