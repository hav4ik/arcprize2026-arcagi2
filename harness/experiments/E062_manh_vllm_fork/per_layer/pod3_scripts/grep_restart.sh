#!/bin/bash
# The restart test's expected lines: cell 2s's "E066 SUPERVISOR:" lines (run.log), its state file, the solver's
# "server died" / "(wait 1): started by the notebook" / "server up after ... (external, start 1)" / re-queue lines,
# the server logs (first server and r<N> relaunches), status.json and attempt errors.
W=/kaggle/working/dsv41_logs
echo "== supervisor (run.log)"; grep -a "E066 SUPERVISOR:" /root/e066/r3/run.log | cut -c1-300 | tail -20
echo "== e066_supervisor.json"; cat $W/e066_supervisor.json 2>/dev/null; echo
echo "== solver.log"; grep -anE "server died|server (is )?down|\(wait [0-9]+\)|server up after|start [0-9]+\)|re-?queue|requeu|retry|restart|status" $W/solver.log | cut -c1-260 | tail -30
echo "== server logs"; for f in $W/server/server.log $W/r*/server/server.log; do [ -f "$f" ] && echo "$f $(stat -c %y $f | cut -c12-19) $(grep -ac 'Application startup complete' $f) startup, $(grep -acE 'Traceback|EngineDead|died|out of memory' $f) errors"; done
echo "== status.json"; cat $W/status.json 2>/dev/null; echo
echo "== attempts: $(wc -l < $W/attempts.jsonl 2>/dev/null) records; errors: $(grep -c '"error"' $W/attempts.jsonl 2>/dev/null)"
