#!/bin/bash
# Dashboard sync for the Manh v12 replica: the server logs' GPU-memory / allocator-cap / throughput lines (start 0, then
# r1, r2, ... in order; original lines) -> data/gpumem.log (overwritten); samp.jsonl -> data/nbruns/m12_rerun/;
# data/ram.tsv: one point per 10 min (utc, kaggle_clock_h, kaggle_eq_gib on the cgroup view [from 03:57 UTC], the
# meminfo form).
D=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data; mkdir -p $D/nbruns/m12_rerun
SSH="timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 18070 root@213.173.105.27"
# every start's server.log in start order: server/ (start 0), then his supervisor's server_restart{n}/ (or our r{n}/server/)
$SSH 'cd /root/kwork_m12_rerun/dsv41_logs && for f in $(find . -name server.log | grep -v traces | awk "{n=0; if (match(\$0, /server_restart[0-9]+/)) n=substr(\$0, RSTART+14, RLENGTH-14); else if (match(\$0, /\\/r[0-9]+\\//)) n=substr(\$0, RSTART+2, RLENGTH-3); print n, \$0}" | sort -n | cut -d" " -f2); do grep -aE "GPU memory:|Capped the PyTorch allocator|Avg generation throughput" $f; done' > $D/gpumem.log.tmp && mv $D/gpumem.log.tmp $D/gpumem.log
$SSH 'head -1 /root/nbruns/m12_rerun/samp.tps.tsv; cat /root/nbruns/m12_rerun/tps_backfill.tsv 2>/dev/null; tail -n +2 /root/nbruns/m12_rerun/samp.tps.tsv | sed -E "s/^([0-9]{2}:[0-9]{2}:[0-9]{2})\t/2026-10-30T\1Z\t/"' > $D/tps.tsv.tmp && mv $D/tps.tsv.tmp $D/tps.tsv
mkdir -p $D/dsv41_logs
$SSH 'cat /root/kwork_m12_rerun/dsv41_logs/attempts.jsonl' > $D/dsv41_logs/attempts.jsonl.tmp && mv $D/dsv41_logs/attempts.jsonl.tmp $D/dsv41_logs/attempts.jsonl
$SSH 'cat /root/kwork_m12_rerun/dsv41_logs/solver.log' > $D/dsv41_logs/solver.log.tmp && mv $D/dsv41_logs/solver.log.tmp $D/dsv41_logs/solver.log
$SSH 'cat /root/nbruns/m12_rerun/samp.jsonl' > $D/nbruns/m12_rerun/samp.jsonl.tmp && mv $D/nbruns/m12_rerun/samp.jsonl.tmp $D/nbruns/m12_rerun/samp.jsonl
python3 - $D <<'PY'
import calendar, json, sys, time
D = sys.argv[1]; T0 = calendar.timegm(time.strptime("2026-10-30 01:51:04", "%Y-%m-%d %H:%M:%S"))
rows, last = [], None
for l in open(f"{D}/nbruns/m12_rerun/samp.jsonl"):
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
echo "gpumem.log $(wc -l < $D/gpumem.log) lines; samp.jsonl $(wc -l < $D/nbruns/m12_rerun/samp.jsonl) samples"
# status.json for the dashboard (computed on the pod from the solver files + the last sampler record)
$SSH 'python3 -I - <<PY
import json, re, time
L = "/kaggle/working/dsv41_logs"; sol = json.load(open("/root/data/evalx2t_s2026_solutions.json"))
rs = [json.loads(l) for l in open(f"{L}/attempts.jsonl")]
ok = [r for r in rs if not r.get("error") and r.get("stop_reason") != "error"]
def correct(r):
    s = sol.get(r["task_id"]); p = int(r.get("pair", 0)); pr = r.get("pred")
    return s is not None and ((len(s) > 1 and isinstance(pr, list) and len(pr) > p and pr[p] == s[p]) or (len(s) == 1 and pr == s[0]))
a0 = [r for r in ok if r.get("attempt") == 0]
answered = {(r["task_id"], r.get("pair", 0)) for r in ok if r.get("pred") is not None}
sl = open(f"{L}/solver.log", errors="replace").read()
last = json.loads(open("/root/nbruns/m12_rerun/samp.jsonl").read().strip().splitlines()[-1])
v = last.get("vllm_gpu_mem") or {}; c = last.get("alloc_cap") or {}; e = last.get("engine_last") or {}; sup = last.get("supervisor") or {}
deaths = sl.count("server died"); restarts = int(sup.get("start") or 0)
retries = max([x["retries"] for x in v.values()] or [0]); ooms = sum(x["ooms"] for x in v.values())
cp = min([x["cap_minus_peak"] for x in c.values() if x.get("cap_minus_peak") is not None] or [None] , key=lambda q: 99 if q is None else q)
keq = last.get("kaggle_eq_gib")
if deaths or restarts or ooms or retries or (len(rs) - len(ok)):
    verdict = f"events: deaths {deaths}, restarts {restarts}, retries {retries}, OOMs {ooms}, errored {len(rs) - len(ok)}"
elif (cp is not None and cp < 1.0) or (keq or 0) > 150:
    verdict = f"approaching limits: cap-peak {cp} GiB, Kaggle-eq RAM {keq} GiB"
else:
    verdict = f"stable: cap-peak {cp} GiB (worst rank), Kaggle-eq RAM {keq} GiB of 173, no events"
T0 = 1793325064.3
st = {"utc": last["utc"], "kaggle_h": round((last["t"] - T0) / 3600, 2), "first_done": len(a0), "first_correct": sum(map(correct, a0)),
      "pairs_answered": len(answered), "errored": len(rs) - len(ok), "deaths": deaths, "restarts": restarts, "retries_max": retries,
      "ooms": ooms, "running": e.get("running"), "waiting": e.get("waiting"), "verdict": verdict}
import os
SF = "/root/nbruns/m12_rerun/status_state.json"
prev = json.load(open(SF)) if os.path.exists(SF) else {}
for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms"):
    st[k] = max(st[k], prev.get(k, 0))
json.dump({k: st[k] for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms")}, open(SF, "w"))
print(json.dumps(st))
PY' > $D/status.json.tmp && mv $D/status.json.tmp $D/status.json && cat $D/status.json
