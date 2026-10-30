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
