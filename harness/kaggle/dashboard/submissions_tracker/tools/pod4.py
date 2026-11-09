"""Third live run, 'Ours V1 qualification' on pod 4. Inactive until its status.json exists."""
import json, os, datetime as dt
R = "/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer"
NAME = "Ours V1 qualification"
DIR = os.environ.get("POD4_DIR", f"{R}/pod4_runs/v1_qual/data")
def info(d=DIR):
    p = f"{d}/status.json"
    if not os.path.exists(p):
        return None
    try:
        st = json.load(open(p)); h = float(st["kaggle_h"]); u = str(st["utc"]).rstrip("Z")
        if "T" in u or "-" in u:
            t = dt.datetime.fromisoformat(u.replace(" ", "T"))
        else:
            now = dt.datetime.utcnow()
            t = dt.datetime.combine(now.date(), dt.time.fromisoformat(u))
            if t > now + dt.timedelta(hours=1): t -= dt.timedelta(days=1)
    except Exception:
        return None
    start = t - dt.timedelta(hours=h)
    return {"t0": start.strftime("%H:%M:%S"), "start": start.strftime("%Y-%m-%d %H:%M"), "dir": d}

# live runs that appear once their status.json exists: name -> config (dir of the pod's synced files, pod label, attempts file, RAM column adjust)
CANDS = {
    NAME: {"dir": DIR, "pod": "pod 4", "att": "attempts.jsonl", "ram_adj": 0.0, "logs": ["gpumem.log"]},
    "SFT 025ep on V12": {"dir": os.environ.get("SFT_DIR", f"{R}/phase_h/sft_025ep_v12/data"), "pod": "pod 2", "att": "dsv41_logs/attempts.jsonl",
                       "ram_adj": -11.3, "logs": ["gpumem.log", "dsv41_logs/server/server.log"]},
    "V12 + images FINAL": {"dir": os.environ.get("IMG_DIR", f"{R}/phase_h/v12_img/data"), "pod": "pod 2", "att": "dsv41_logs/attempts.jsonl",
                     "ram_adj": -11.3, "logs": ["gpumem.log", "dsv41_logs/server/server.log"]},
}
# phase_h dirs that are already handled (frozen, hard-coded or retired): never auto-discovered
KNOWN_DIRS = {"manh_v12", "v12_img_r0", "v12_img_ab_r1", "v12_img_rg_bad", "sft_025ep_v12", "v12_img", "ab_decode", "runs", "scripts"}
LIVE_MAX_AGE_S = 3 * 3600  # an unmapped dir counts as a live run while its status.json is newer than this
def discover():
    """Any phase_h/<dir>/data/status.json not in CANDS/KNOWN_DIRS and fresh: label = status.json 'run'/'label'/'name' field, else the dir name."""
    import glob, time
    out = {}
    mapped = {c["dir"] for c in CANDS.values()}
    for p in sorted(glob.glob(f"{R}/phase_h/*/data/status.json")):
        d = os.path.dirname(p)
        if d in mapped or d.split("/")[-2] in KNOWN_DIRS or time.time() - os.path.getmtime(p) > LIVE_MAX_AGE_S:
            continue
        try: st = json.load(open(p))
        except Exception: continue
        name = str(st.get("run") or st.get("label") or st.get("name") or d.split("/")[-2])
        out[name] = {"dir": d, "pod": str(st.get("pod") or "pod ?"), "att": "dsv41_logs/attempts.jsonl", "ram_adj": float(st.get("ram_adj", -11.3)),
                     "logs": ["gpumem.log", "dsv41_logs/server/server.log"]}
    return out
def active():
    """name -> info dict (t0, start, dir, pod, att path, ram_adj, logs) for every candidate whose status.json exists."""
    out = {}
    for name, c in {**discover(), **CANDS}.items():
        i = info(c["dir"])
        if i:
            out[name] = dict(i, pod=c["pod"], att=f"{c['dir']}/{c['att']}", ram_adj=c["ram_adj"], logs=[f"{c['dir']}/{l}" for l in c["logs"]])
    return out
