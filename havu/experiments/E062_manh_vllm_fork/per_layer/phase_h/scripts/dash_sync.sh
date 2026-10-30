#!/bin/bash
# Dashboard sync for the Manh v12 replica: the server logs' GPU-memory / allocator-cap / throughput lines (start 0, then
# r1, r2, ... in order; original lines) -> data/gpumem.log (overwritten); samp.jsonl -> data/nbruns/m12_rerun/;
# data/ram.tsv: one point per 10 min (utc, kaggle_clock_h, kaggle_eq_gib on the cgroup view [from 03:57 UTC], the
# meminfo form).
D=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data; mkdir -p $D/nbruns/m12_rerun
SSH="timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 18070 root@213.173.105.27"
$SSH 'cd /root/kwork_m12_rerun/dsv41_logs && for f in server/server.log $(ls -d r[0-9]*/server/server.log 2>/dev/null | sort -V); do grep -aE "GPU memory:|Capped the PyTorch allocator|Avg generation throughput" $f; done' > $D/gpumem.log.tmp && mv $D/gpumem.log.tmp $D/gpumem.log
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
