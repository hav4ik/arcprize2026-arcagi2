#!/bin/bash
# Dashboard sync for a replica run (REPLICA_RUN = run name on pod 2, REPLICA_DATA = local data dir; dash_sync.sh generalised; same files and
# formats as manh_v12/data): every start's server.log GPU-memory / allocator-cap / throughput lines -> data/gpumem.log;
# samp.jsonl -> data/nbruns/$RUN/; samp.tps.tsv -> data/tps.tsv (ISO utc); data/ram.tsv one point per 10 min;
# attempts.jsonl, solver.log, metrics.jsonl -> data/dsv41_logs/; status.json computed on the pod. T0 = the launch (notebook.sha256 mtime).
RUN=${REPLICA_RUN:?}; D=${REPLICA_DATA:?}
mkdir -p $D/nbruns/$RUN $D/dsv41_logs
SSH="timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 29668 root@157.157.221.29"
T0=$($SSH "python3 -c \"import os; print(os.path.getmtime('/root/nbruns/$RUN/notebook.sha256'))\"") || exit 1
$SSH "cd /root/kwork_$RUN/dsv41_logs && for f in \$(find . -name server.log | grep -v traces | awk '{n=0; if (match(\$0, /server_restart[0-9]+/)) n=substr(\$0, RSTART+14, RLENGTH-14); else if (match(\$0, /\\/r[0-9]+\\//)) n=substr(\$0, RSTART+2, RLENGTH-3); print n, \$0}' | sort -n | cut -d' ' -f2); do grep -aE 'GPU memory:|Capped the PyTorch allocator|Avg generation throughput' \$f; done" > $D/gpumem.log.tmp && mv $D/gpumem.log.tmp $D/gpumem.log
DAY=$(date -u +%Y-%m-%d)
$SSH "head -1 /root/nbruns/$RUN/samp.tps.tsv; tail -n +2 /root/nbruns/$RUN/samp.tps.tsv" | python3 -c "
import sys, time
t0 = $T0; rows = sys.stdin.read().splitlines()
print(rows[0] if rows else 'utc\tkaggle_h\tgen_tok_s\trunning\twaiting\tkv_pct')
for r in rows[1:]:
    p = r.split('\t'); kh = float(p[1]); t = t0 + kh * 3600
    print(time.strftime('%Y-%m-%dT', time.gmtime(t)) + p[0] + 'Z\t' + '\t'.join(p[1:]))
" > $D/tps.tsv.tmp && mv $D/tps.tsv.tmp $D/tps.tsv
$SSH "cat /root/kwork_$RUN/dsv41_logs/attempts.jsonl 2>/dev/null" > $D/dsv41_logs/attempts.jsonl.tmp && mv $D/dsv41_logs/attempts.jsonl.tmp $D/dsv41_logs/attempts.jsonl
$SSH "cat /root/kwork_$RUN/dsv41_logs/metrics.jsonl 2>/dev/null" > $D/dsv41_logs/metrics.jsonl.tmp && mv $D/dsv41_logs/metrics.jsonl.tmp $D/dsv41_logs/metrics.jsonl
$SSH "cat /root/kwork_$RUN/dsv41_logs/solver.log 2>/dev/null" > $D/dsv41_logs/solver.log.tmp && mv $D/dsv41_logs/solver.log.tmp $D/dsv41_logs/solver.log
$SSH "cat /root/nbruns/$RUN/samp.jsonl" > $D/nbruns/$RUN/samp.jsonl.tmp && mv $D/nbruns/$RUN/samp.jsonl.tmp $D/nbruns/$RUN/samp.jsonl
python3 - $D $RUN $T0 <<'PY'
import json, sys, time
D, RUN, T0 = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows, last = [], None
for l in open(f"{D}/nbruns/{RUN}/samp.jsonl"):
    r = json.loads(l)
    if "meminfo_gib" not in r: continue
    if last is not None and r["t"] - last < 600: continue
    last = r["t"]
    cg = r.get("kaggle_eq_gib") if r.get("cgroup_gib") else ""
    mi = r.get("kaggle_eq_meminfo_gib") or round(r["meminfo_gib"]["AnonPages"] + r["meminfo_gib"]["Shmem"] + 7, 2)
    rows.append(f"{r['utc']}\t{(r['t'] - T0) / 3600:.2f}\t{cg}\t{mi}")
open(f"{D}/ram.tsv", "w").write("utc\tkaggle_clock_h\tkaggle_eq_gib_cgroup\tkaggle_eq_gib_meminfo_form\n" + "\n".join(rows) + "\n")
print(f"ram.tsv {len(rows)} points")
PY
echo "gpumem.log $(wc -l < $D/gpumem.log) lines; samp.jsonl $(wc -l < $D/nbruns/$RUN/samp.jsonl) samples"
$SSH "python3 -I - $RUN $T0" <<'PY' > $D/status.json.tmp && mv $D/status.json.tmp $D/status.json && cat $D/status.json
import json, os, sys
RUN, T0 = sys.argv[1], float(sys.argv[2])
L = f"/root/kwork_{RUN}/dsv41_logs"; sol = json.load(open("/root/data/evalx2t_s2026_solutions.json"))
rs = [json.loads(l) for l in open(f"{L}/attempts.jsonl")] if os.path.exists(f"{L}/attempts.jsonl") else []
ok = [r for r in rs if not r.get("error") and r.get("stop_reason") != "error"]
def correct(r):
    s = sol.get(r["task_id"]); p = int(r.get("pair", 0)); pr = r.get("pred")
    return s is not None and ((len(s) > 1 and isinstance(pr, list) and len(pr) > p and pr[p] == s[p]) or (len(s) == 1 and pr == s[0]))
a0 = [r for r in ok if r.get("attempt") == 0]
answered = {(r["task_id"], r.get("pair", 0)) for r in ok if r.get("pred") is not None}
sl = open(f"{L}/solver.log", errors="replace").read() if os.path.exists(f"{L}/solver.log") else ""
last = json.loads(open(f"/root/nbruns/{RUN}/samp.jsonl").read().strip().splitlines()[-1])
v = last.get("vllm_gpu_mem") or {}; c = last.get("alloc_cap") or {}; e = last.get("engine_last") or {}; sup = last.get("supervisor") or {}
deaths = sl.count("server died"); restarts = int(sup.get("start") or 0)
retries = max([x["retries"] for x in v.values()] or [0]); ooms = sum(x["ooms"] for x in v.values())
cp = min([x["cap_minus_peak"] for x in c.values() if x.get("cap_minus_peak") is not None] or [None], key=lambda q: 99 if q is None else q)
cr = min([x["cap_minus_reserved_now"] for x in c.values() if x.get("cap_minus_reserved_now") is not None] or [None], key=lambda q: 99 if q is None else q)
keq = last.get("kaggle_eq_gib")
if deaths or restarts or ooms or retries or (len(rs) - len(ok)):
    verdict = f"events: deaths {deaths}, restarts {restarts}, retries {retries}, OOMs {ooms}, errored {len(rs) - len(ok)}"
elif (cr is not None and cr < 0.30) or (keq or 0) > 150:
    verdict = f"approaching limits: cap-reserved {cr} GiB, Kaggle-eq RAM {keq} GiB"
else:
    verdict = f"stable: cap-reserved {cr} GiB, cap-peak {cp} GiB (worst rank), Kaggle-eq RAM {keq} GiB of 173, no events"
st = {"utc": last["utc"], "kaggle_h": round((last["t"] - T0) / 3600, 2), "first_done": len(a0), "first_correct": sum(map(correct, a0)),
      "pairs_answered": len(answered), "errored": len(rs) - len(ok), "deaths": deaths, "restarts": restarts, "retries_max": retries,
      "ooms": ooms, "running": e.get("running"), "waiting": e.get("waiting"), "verdict": verdict}
SF = f"/root/nbruns/{RUN}/status_state.json"
prev = json.load(open(SF)) if os.path.exists(SF) else {}
for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms"):
    st[k] = max(st[k], prev.get(k, 0))
json.dump({k: st[k] for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms")}, open(SF, "w"))
print(json.dumps(st))
PY
