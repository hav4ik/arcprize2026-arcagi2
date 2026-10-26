#!/bin/bash
# summ.sh <run> <challenges>: notebook outcome, key solver events, validation
A=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_e2e
R=$A/runs/$1
echo "######## $1 (alive: $(kill -0 $(cat $R/nb.pid) 2>/dev/null && echo yes || echo no))"
grep -E "E046 (CHECK|FALLBACK)|kill switch|solver exit code|submission.json (valid|invalid)|RUN_NB (notebook|cell [0-9]+ RAISED)|solver cell failed|Traceback" $R/run.log | cut -c1-220
grep -E "server (start|died|up|failed|not up|down)|exited during|force deadline|hard stop|admission closed|KV gate|crash|result handling failed|write failed|signal|warning|fallback submission|done$" $R/work/dsv41_logs/solver.log 2>/dev/null | cut -c1-230 | sed 's/model \/tmp[^ ]*/model <fake>/'
/home/vu/Workspace/ARC-AGI-2/.venv/bin/python $A/tools/validate.py $2 $R/work/submission.json $R/work/dsv41_logs 2>&1 | tail -9
