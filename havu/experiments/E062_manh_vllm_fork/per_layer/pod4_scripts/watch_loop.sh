#!/bin/bash
# Local watch loop for a pod 4 run (read-only on the pod). Every 60 s: the kit's watcher --once over ssh (own state
# file poll.json, alerts poll.alerts). Every 5 min: export_dash.sh (dashboard files), plus checks on status.json that
# the kit's watcher does not make: cap - reserved < 0.30 GiB (the binding VRAM margin, pod 3's OOM lesson), new
# finalized errors, new deaths, supervisor not up. Exits (so the caller is woken) on the first watcher level >= ALERT, any
# status flag, the notebook's end, or at UTC HH:MM (argv 2: the next report time). WARN lines are collected and printed
# at exit. Usage: watch_loop.sh <run> <HH:MM> [--quiet-alerts]   (--quiet-alerts: do not exit on watcher ALERTs, e.g.
# during a planned fault; status flags still exit)
S=$(cd "$(dirname "$0")" && pwd); read IP PORT < $S/pod4_ssh.txt; RUN=$1; UNTIL=$2; QUIET=${3:-}
O=$S/../pod4_runs/$RUN/data; ST=$S/../pod4_runs/$RUN/watch_local.json; WARN=$(mktemp)
SSH=(timeout 90 ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$IP)
R=/root/runs/$RUN
end_epoch=$(date -u -d "$(date -u +%F) $UNTIL" +%s); [ $end_epoch -lt $(date +%s) ] && end_epoch=$((end_epoch + 86400))
last_dash=0
finish() { echo "== $1"; [ -s $WARN ] && { echo "-- warnings since last report:"; sort -u $WARN | tail -20; }; echo "-- status: $(cat $O/status.json 2>/dev/null)"; rm -f $WARN; exit 0; }
while true; do
  out=$("${SSH[@]}" "/usr/bin/python3 /root/kit/watcher.py --samples $R/mon.jsonl --once --summary --state $R/poll.json --alerts $R/poll.alerts" 2>&1); rc=$?
  echo "$out" | grep -E " (WARN|ALERT|CRIT) " >> $WARN
  if [ $rc -ge 3 ] && [ $rc -le 4 ] && [ "$QUIET" != --quiet-alerts ]; then $S/export_dash.sh $RUN > /dev/null 2>&1; finish "WATCHER ALERT (rc $rc): $(echo "$out" | grep -E ' (ALERT|CRIT) ' | head -8 | tr '\n' ' ')"; fi
  if [ $(( $(date +%s) - last_dash )) -ge 290 ]; then
    $S/export_dash.sh $RUN > /dev/null 2>&1; last_dash=$(date +%s)
    flag=$(python3 - $O/status.json $ST <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
try:
    p = json.load(open(sys.argv[2]))
except Exception:
    p = {}
f = []
cr = s.get("cap_minus_reserved_min")
if cr is not None and cr < 0.30 and not p.get("cr_flagged"):
    f.append(f"cap - reserved {cr} GiB < 0.30"); p["cr_flagged"] = True
for k in ("finalized_errors", "deaths", "ooms", "retries_max", "timeouts"):
    if (s.get(k) or 0) > p.get(k, 0):
        f.append(f"{k} {p.get(k, 0)} -> {s.get(k)}")
    p[k] = max(p.get(k, 0), s.get(k) or 0)
sup = s.get("supervisor")
if sup not in (None, "up") and p.get("sup") != sup:
    f.append(f"supervisor {sup}")
p["sup"] = sup
json.dump(p, open(sys.argv[2], "w"))
print("; ".join(f))
PY
)
    [ -n "$flag" ] && finish "STATUS FLAG: $flag"
  fi
  "${SSH[@]}" "grep -q NOTEBOOK_DONE $R/nb.timeline" 2>/dev/null && { $S/export_dash.sh $RUN > /dev/null 2>&1; finish "NOTEBOOK END: $("${SSH[@]}" "grep NOTEBOOK_DONE $R/nb.timeline")"; }
  [ $(date +%s) -ge $end_epoch ] && { $S/export_dash.sh $RUN > /dev/null 2>&1; finish "REPORT DUE ($UNTIL UTC); watcher: $(echo "$out" | tail -4 | tr '\n' ' ')"; }
  sleep 60
done
