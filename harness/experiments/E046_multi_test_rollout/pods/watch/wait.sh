#!/bin/bash
# wait up to $1 s for a new alert or a completed poll; print new alerts and the new rows
P=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/podwatch
a0=$(wc -l < $P/alerts.log); r0=$(wc -l < $P/loop2.log); p0=$(grep -c POLLDONE $P/loop2.log); end=$(( $(date +%s) + ${1:-580} ))
while [ $(date +%s) -lt $end ]; do
  [ $(wc -l < $P/alerts.log) -gt $a0 ] && break
  [ $(grep -c POLLDONE $P/loop2.log) -gt $p0 ] && break
  sleep 5
done
tail -n +$((a0 + 1)) $P/alerts.log
tail -n +$((r0 + 1)) $P/loop2.log | grep ROW | cut -d'|' -f2-16
ps -p $(cat $P/alert.pid) > /dev/null || echo "ALERT LOOP DEAD"; ps -p $(cat $P/loop2.pid) > /dev/null || echo "POLL LOOP DEAD"
date -u +%T
