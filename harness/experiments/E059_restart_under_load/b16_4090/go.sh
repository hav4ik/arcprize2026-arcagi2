#!/usr/bin/env bash
# E059 B16 box 57983124: the bring-up chain, then (only on CHAIN OK, i.e. every verify gate passed and the GPUs are idle)
# the B16 restart-under-load run.
#   setsid -f bash /root/e059/scripts/go.sh > /root/e059/runs/go.driver.out 2>&1 < /dev/null
bash /root/e059/scripts/setup_chain.sh > /root/e059/runs/setup_chain.driver.out 2>&1
if tail -3 /root/e059/runs/setup_chain.log | grep -q "CHAIN OK"; then
  bash /root/e059/scripts/run_b16.sh e059_b16kill 5400 > /root/e059/runs/run_b16.driver.out 2>&1
else
  echo "$(date -u +%FT%TZ) setup chain did not end OK: B16 run not started" >> /root/e059/runs/e059_b16kill.progress
fi
