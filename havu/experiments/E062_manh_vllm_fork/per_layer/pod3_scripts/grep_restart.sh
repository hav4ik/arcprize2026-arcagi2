#!/bin/bash
# The restart test's expected lines: solver ("server died", "(wait 1): started by the notebook", "server up after ...
# (external, start 1)", re-queued attempts), the supervisor's lines in the server dir, the status file(s), and run.log.
W=/kaggle/working/dsv41_logs
echo "== solver.log"; grep -anE "server died|server (is )?down|\(wait [0-9]+\)|server up after|start [0-9]+\)|re-?queue|requeu|retry|restart" $W/solver.log | cut -c1-260 | tail -30
echo "== supervisor / server dir"; ls -la --time-style=+%T $W/server/ 2>/dev/null
grep -ahiE "supervis|restart|exit(ed)? with|died|relaunch|status" $W/server/*.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -viE "GET /metrics" | cut -c1-260 | tail -30
echo "== status files"; for f in $W/server/*status* ${ARC_EXTERNAL_SERVER_STATUS:-}; do [ -f "$f" ] && { echo "-- $f"; cat "$f"; echo; }; done
echo "== run.log"; grep -aE "MANH|supervis|restart|server" /root/e066/r3/run.log | cut -c1-260 | tail -15
echo "== status.json"; cat $W/status.json 2>/dev/null; echo
echo "== attempts: $(wc -l < $W/attempts.jsonl 2>/dev/null) records; errors: $(grep -c '"error"' $W/attempts.jsonl 2>/dev/null)"
