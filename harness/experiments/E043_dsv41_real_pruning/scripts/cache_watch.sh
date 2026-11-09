#!/bin/bash
# E043: watch the prefix-cache health of the active eval servers on the 8x box (the newest runs/tp4A_* and tp4B_*).
# Every 2 min: running / waiting / KV usage / prefix hit rate from vLLM's 10 s stats line, plus preemption counts.
# Appends to results/cache_watch.tsv and exits (so the caller is notified) when KV usage > 85%, the hit rate < 60%
# after a run's first 30 min, or a preemption appears.   Usage: cache_watch.sh
OUT=$(dirname "$0")/../results/cache_watch.tsv
[ -s $OUT ] || echo -e "utc\trun\trunning\twaiting\tkv_pct\thit_pct\tpreemptions\tminutes_up" > $OUT
while true; do
  R=$(timeout 60 ssh -o BatchMode=yes -p 50293 root@147.185.60.21 'cd /workspace/dsv4/runs; for s in A B; do d=$(ls -td tp4${s}_*/ 2>/dev/null | head -1); d=${d%/}; [ -n "$d" ] || continue; l=$(grep "Engine 000" $d/server.log | tail -1); p=$(grep -ci preempt $d/server.log); up=$(( ( $(date +%s) - $(stat -c %Y $d/server.pid) ) / 60 )); echo "$d|$l|$p|$up"; done' 2>/dev/null | grep -v "^Welcome\|^Have fun\|^AI agents")
  ALERT=""
  while IFS='|' read -r run line pre up; do
    [ -n "$line" ] || continue
    read -r rn wt kv hit <<< "$(echo "$line" | sed -E 's/.*Running: ([0-9]+) reqs, Waiting: ([0-9]+) reqs, GPU KV cache usage: ([0-9.]+)%, Prefix cache hit rate: ([0-9.]+)%.*/\1 \2 \3 \4/')"
    echo -e "$(date -u +%FT%TZ)\t$run\t$rn\t$wt\t$kv\t$hit\t$pre\t$up" >> $OUT
    awk -v k="$kv" 'BEGIN{exit !(k>85)}' && ALERT="$ALERT $run: KV ${kv}%;"
    [ "$up" -gt 30 ] && [ "$rn" -gt 20 ] && awk -v h="$hit" 'BEGIN{exit !(h<60)}' && ALERT="$ALERT $run: hit ${hit}% at ${rn} running;"
    [ "${pre:-0}" -gt 0 ] && ALERT="$ALERT $run: ${pre} preemption lines;"
  done <<< "$R"
  [ -n "$ALERT" ] && { echo "ALERT $(date -u +%T):$ALERT"; tail -4 $OUT; exit 0; }
  sleep 120
done
