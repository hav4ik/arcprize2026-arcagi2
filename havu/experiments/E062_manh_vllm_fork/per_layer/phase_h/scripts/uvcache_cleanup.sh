#!/bin/bash
# Pod-only accommodation (as in the V12 replica): once the install cells are done (nb.timeline "END   cell 11", Manh's
# cell 5), delete the ~20 GB uv cache /tmp/dsv41/cache/uv for disk headroom. Nothing reads it afterwards.
R=$1; t0=$(date +%s)
until grep -q "END   cell 11:" $R/nb.timeline 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "$(date -u +%T) gave up waiting"; exit 1; }
  grep -q NOTEBOOK_DONE $R/nb.timeline 2>/dev/null && { echo "$(date -u +%T) notebook ended before cell 11"; exit 1; }
  sleep 5
done
echo "$(date -u +%T) cell 11 done; uv cache $(du -sh /tmp/dsv41/cache/uv 2>/dev/null | cut -f1); free before: $(df -h / | tail -1 | awk '{print $4}')"
rm -rf /tmp/dsv41/cache/uv
echo "$(date -u +%T) deleted; free after: $(df -h / | tail -1 | awk '{print $4}')"
