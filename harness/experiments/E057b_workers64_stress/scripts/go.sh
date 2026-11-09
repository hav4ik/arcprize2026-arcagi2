#!/usr/bin/env bash
# E057b: resume the setup chain at step 6 (download checks, 55% memory gate, one B+ verify start); on CHAIN OK start
# the stress chain (A = 56 workers, then B = 64 workers, 16 GiB offload).
FROM=6 bash /root/e057b/scripts/setup_chain.sh > /root/e057b/runs/setup_chain.driver3.out 2>&1
if tail -3 /root/e057b/runs/setup_chain.log | grep -q "CHAIN OK"; then
  bash /root/e057b/scripts/stress_chain.sh > /root/e057b/runs/stress_chain.driver.out 2>&1
else
  echo "$(date -u +%FT%TZ) setup chain did not end OK: stress chain not started" >> /root/e057b/runs/stress_chain.log
fi
