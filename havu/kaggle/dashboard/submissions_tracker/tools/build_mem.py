"""Rebuild the dashboard's mem.json: stopped runs as recorded, live runs as dense series.
GPU: from the server log's per-rank 'GPU memory:' lines (lowest vLLM-reported free and cap minus peak over ranks, 10-min
bins). RAM: Kaggle-eq from the pods' samplers (container view), thinned to 10 min."""
import json, os, subprocess, sys
D = os.environ.get("DASH_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the dashboard folder (live copy: the session scratchpad)
R = "/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer"
LIVE = {
    "Manh V12 replica": {"t0": "01:56:00", "logs": [f"{R}/phase_h/manh_v12/data/gpumem.log", f"{R}/phase_h/manh_v12/data/dsv41_logs/server/server.log"],
                         "ram": ("tsv", f"{R}/phase_h/manh_v12/data/ram.tsv", -11.3)},
    "B32-M V3 replica": {"t0": "03:30:40", "logs": [f"{R}/pod3_runs/b32m_v3/data/gpumem.log"],
                         "ram": ("tsv", f"{R}/pod3_runs/b32m_v3/data/ram.tsv", 0.0)},
}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pod4
P4 = pod4.info()
if P4:
    LIVE[pod4.NAME] = {"t0": P4["t0"], "logs": [f"{P4['dir']}/gpumem.log"], "ram": ("tsv", f"{P4['dir']}/ram.tsv", 0.0)}
old = json.load(open(f"{D}/mem.json"))
out = [r for r in old if r["run"] not in LIVE]
extra = json.load(open(f"{D}/mem_live_points.json")) if os.path.exists(f"{D}/mem_live_points.json") else []
def hms(s):
    h, m, *sec = map(int, s.split(":")); return h * 3600 + m * 60 + (sec[0] if sec else 0)
for run, cfg in LIVE.items():
    log = next((p for p in cfg["logs"] if os.path.exists(p)), None)
    gpu_max_t = -1
    if log:
        res = subprocess.run([sys.executable, f"{D}/tools/gpumem_series.py", log, cfg["t0"], "10"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            th, cp, free, retr, oom, utc, cr = line.split("\t")
            out.append({"run": run, "t_h": float(th), "ram_kaggle_eq_gib": None, "vram_free_min_gib": float(free),
                        "cap_minus_peak_gib": float(cp), "cap_minus_reserved_gib": float(cr)})
            gpu_max_t = max(gpu_max_t, float(th))
    kind, path, adj = cfg["ram"]
    ram_max_t = -1
    if os.path.exists(path):
        t0 = hms(cfg["t0"]); last_bin = None
        if kind == "samp":
            pts = [(hms(r["utc"]), r["kaggle_eq_gib"]) for r in map(json.loads, filter(str.strip, open(path))) if r.get("kaggle_eq_gib")]
        else:
            # column 2 is the container (cgroup) view; where it is empty, column 3 is the host-wide meminfo form + adj
            pts = [(t0 + 3600 * float(c[1]), float(c[2]) if c[2].strip() else float(c[3]) + adj)
                   for c in (l.rstrip("\n").split("\t") for l in open(path) if l[0].isdigit())]
        for t, v in pts:
            th = (t - t0) / 3600
            low = th > 1.0 and v < 50  # a real dip (e.g. the server restarting) stays
            if th < 0.1 or (v < 50 and not low): continue
            b = int(th * 6)
            if b == last_bin and not low: continue
            last_bin = b
            out.append({"run": run, "t_h": round(th, 2), "ram_kaggle_eq_gib": round(v + (adj if kind == "samp" else 0), 1), "vram_free_min_gib": None, "cap_minus_peak_gib": None})
            ram_max_t = max(ram_max_t, th)
    # report points newer than the synced data (from pod messages)
    for p in extra:
        if p["run"] == run:
            q = dict(p)
            if q["t_h"] <= gpu_max_t: q["vram_free_min_gib"] = q["cap_minus_peak_gib"] = None
            if q["t_h"] <= ram_max_t: q["ram_kaggle_eq_gib"] = None
            if any(q[k] is not None for k in ("vram_free_min_gib", "cap_minus_peak_gib", "ram_kaggle_eq_gib")):
                out.append(q)
out.sort(key=lambda r: (r["run"], r["t_h"]))
json.dump(out, open(f"{D}/mem.json", "w"), indent=1)
for run in LIVE:
    rs = [r for r in out if r["run"] == run]
    print(run, len(rs), "gpu pts", sum(r["vram_free_min_gib"] is not None for r in rs), "ram pts", sum(r["ram_kaggle_eq_gib"] is not None for r in rs))
