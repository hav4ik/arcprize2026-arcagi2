#!/bin/bash
# Phase 2 checkpoint report (read-only): split table, attempts, gate holds, errors, host RAM, GPU memory, disk.
# Usage: p2_report.sh [run dir=/root/nbruns/p2_rerun] [KV pool=2171851]
W=/kaggle/working; L=$W/dsv41_logs; R=${1:-/root/nbruns/p2_rerun}; POOL=${2:-2171851}; S=$L/server/server.log
echo "== $(date -u +%T) UTC; notebook: $(grep -a START $R/nb.timeline | tail -1 | cut -c1-90)"
grep -a "NOTEBOOK_DONE" $R/nb.timeline
echo "== split table (pool $POOL)"; python3 -I /root/e062pl/windows2.py $L/metrics.jsonl $POOL 2>&1 | tail -n 40
echo "== attempts"; python3 -I - "$L/attempts.jsonl" /root/data/evalx2t_s2026_solutions.json <<'PY'
import collections, json, sys
sol = json.load(open(sys.argv[2])); recs = [json.loads(l) for l in open(sys.argv[1])]
stops = collections.Counter(r.get("stop_reason") for r in recs); errs = collections.Counter(str(r.get("error"))[:80] for r in recs if r.get("error"))
def ok(r):  # as first_correct.py: multi-test records carry the list of all test outputs in pred
    t, p, pr = r["task_id"], int(r.get("pair", 0)), r.get("pred"); s = sol.get(t)
    if pr is None or s is None or r.get("error"):
        return False
    return (len(s) > 1 and isinstance(pr, list) and len(pr) > p and pr[p] == s[p]) or (len(s) == 1 and pr == s[0])
pairs = {}
for r in recs:
    k = (r["task_id"], int(r.get("pair", 0))); pairs.setdefault(k, []).append(r)
a0 = [r for r in recs if r.get("attempt") == 0]
print(f"attempt records {len(recs)}, correct {sum(map(ok, recs))} ({sum(map(ok, recs)) / max(1, len(recs)):.3f}); "
      f"with pred {sum(r.get('pred') is not None for r in recs)}; stops {dict(stops)}")
print(f"first attempts {len(a0)}, correct {sum(map(ok, a0))} ({sum(map(ok, a0)) / max(1, len(a0)):.3f}); "
      f"pairs touched {len(pairs)} of 344; tasks {len({k[0] for k in pairs})} of 240; "
      f"pairs with any correct {sum(any(map(ok, v)) for v in pairs.values())}")
print("errors:", dict(errs) if errs else 0)
PY
echo "== gate holds: $(grep -aic 'gate' $L/solver.log) solver.log lines; last: $(grep -ai 'gate' $L/solver.log | tail -1 | cut -c1-160)"
echo "== server: HTTP 4xx/5xx $(grep -acE 'HTTP/1.1" [45][0-9][0-9]' $S); ERROR lines $(grep -ac ' ERROR ' $S); Traceback $(grep -ac Traceback $S); preempt lines $(grep -aci preempt $S)"
echo "== solver last metrics line: $(grep -a 'server:' $L/solver.log | tail -1 | cut -c1-300)"
echo "== host RAM: now $(free -g | awk 'NR==2{print "used "$3" GiB, avail "$7" GiB"}'); peak used $(awk -F'used_gib=' '{split($2,a," "); if (a[1]>m) m=a[1]} END {print m}' $R/mem.log) GiB (mem.log, 60 s)"
echo "== GPU: $(nvidia-smi --query-gpu=index,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader | paste -sd';')"
echo "== GPU memory lines (last per rank):"; sed 's/\x1b\[[0-9;]*m//g' $S | grep -a 'GPU memory' | tail -4 | cut -c1-260
echo "== supervisor: $(tr -d '\n' < $L/e066_supervisor.json 2>/dev/null | cut -c1-400)"
echo "== solver server-side/re-queue lines: $(grep -aciE 'server-side|re-?queue' $L/solver.log); attempt records with an error: $(grep -ac '"error": "' $L/attempts.jsonl)"
echo "== disk free: $(df -BG --output=avail / | tail -1) ; /dev/shm $(df -BG --output=avail /dev/shm | tail -1)"
