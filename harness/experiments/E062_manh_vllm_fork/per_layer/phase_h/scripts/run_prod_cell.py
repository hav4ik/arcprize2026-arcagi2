"""Run Manh's PRODUCTION server cell (cell 14 "# Cell 2m (B32-M + Manh's server)" of
scratchpad/manh_ext_nb/latest/dsv41-tools-submit-b32m-external-server.ipynb) verbatim on the pod, with the notebook's
globals set to the pod's paths. The cell launches dsv41_vllm.py, waits for /v1/models, and defines MANH_SERVER_LOG.
Usage: MANH_VISION=1 MANH_VISION_MAX_IMAGES=32 run_prod_cell.py <cell.py> <work dir> <model dir> [extra --vllm-arg ...]
Extra args (DCP4 variant) are appended to the launcher argv by wrapping subprocess.run (the cell text stays verbatim)."""
import json, os, subprocess, sys, time
cell, work, model = sys.argv[1:4]
extra = sys.argv[4:]
_run = subprocess.run
def run_with_extra(cmd, *a, **k):
    if extra and isinstance(cmd, list) and any(str(x).endswith("dsv41_vllm.py") for x in cmd):
        cmd = list(cmd) + extra
        print("EXTRA LAUNCHER ARGS", extra, flush=True)
    return _run(cmd, *a, **k)
subprocess.run = run_with_extra
g = {"__name__": "prod_cell", "json": json, "os": os, "time": time, "subprocess": subprocess,
     "MODE": "submit", "RERUN": False, "MODEL_SRC": model, "WORK": work, "STAGE_DONE": "",
     "BASE_ENV": dict(os.environ), "PY": sys.executable, "T0": time.time()}
exec(compile(open(cell).read(), "manh_prod_cell", "exec"), g)
open(os.path.join(work, "last_prod_server.json"), "w").write(json.dumps(
    {"MANH_SERVER_LOG": g.get("MANH_SERVER_LOG"), "MANH_SERVER_PID": g.get("MANH_SERVER_PID"),
     "MANH_BASE_URL": g.get("MANH_BASE_URL"), "MODE": g.get("MODE")}))
print("PROD_CELL_DONE", g.get("MANH_SERVER_LOG"), g.get("MANH_SERVER_PID"), flush=True)
