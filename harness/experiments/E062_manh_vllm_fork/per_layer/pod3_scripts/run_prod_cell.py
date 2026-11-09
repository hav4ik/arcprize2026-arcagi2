"""Run Manh's production server cell (his new notebook's cell 2m, verbatim) on the pod with the notebook's globals:
MODEL_SRC, WORK (arc2-2026-src copied as his cell 2 does), STAGE_DONE "" , BASE_ENV, PY, T0, MODE, RERUN.
Usage (venv python): run_prod_cell.py <cell.py> <work> <model dir> <out json>
Writes <out json>: MANH_SERVER_PID, MANH_SERVER_LOG, MANH_BASE_URL, MANH_MODEL, wall seconds."""
import json, os, subprocess, sys, time
cell, work, model, out = sys.argv[1:5]
os.environ.setdefault("MANH_VISION", "1"); os.environ.setdefault("MANH_VISION_MAX_IMAGES", "32")
os.environ.setdefault("ARC_PORT", "18084"); os.environ.setdefault("ARC_CONTEXT_LIMIT", "163840")
base = {k: v for k, v in os.environ.items()}
base.update(PATH="/root/b32m/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
            CUDA_HOME="/usr/local/cuda-13.0")
g = {"os": os, "json": json, "time": time, "subprocess": subprocess, "sys": sys, "MODE": "submit", "RERUN": False,
     "MODEL_SRC": model, "WORK": work, "STAGE_DONE": "", "BASE_ENV": base, "PY": "/root/b32m/venv/bin/python",
     "T0": time.time(), "__name__": "cell2m"}
src = open(cell).read()
extra = json.loads(os.environ.get("EXTRA_VLLM_ARGS", "[]"))  # e.g. the DCP4 variant; appended after his last flag
if extra:
    anchor = '"--vllm-arg=--prefix-cache-retention-interval=128"]'
    assert src.count(anchor) == 1, "anchor not found once in his cell"
    src = src.replace(anchor, '"--vllm-arg=--prefix-cache-retention-interval=128", ' + ", ".join(json.dumps(a) for a in extra) + "]")
    print("EXTRA_VLLM_ARGS", extra, flush=True)
t = time.time()
exec(compile(src, "manh_cell2m", "exec"), g)
res = {k: g.get(k) for k in ("MANH_SERVER_PID", "MANH_SERVER_LOG", "MANH_BASE_URL", "MANH_MODEL", "MODE")}
res["wall_s"] = round(time.time() - t)
json.dump(res, open(out, "w"), indent=1)
print("CELL2M_RESULT", json.dumps(res), flush=True)
