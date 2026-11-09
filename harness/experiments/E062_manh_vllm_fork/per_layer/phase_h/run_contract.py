"""Run kaggle/dsv41_tools/cells/b32m_contract.py's contract_run (unchanged) against an external server.
Usage: run_contract.py <base url> <kit dir> <out json>"""
import json, os, sys, time
base, kit, out = sys.argv[1:4]
os.environ.setdefault("ARC_GRID_FORMAT", "text+image")
os.environ.setdefault("ARC_CONTEXT_LIMIT", "163840")
g = {"WORK": os.path.dirname(out), "T0": time.time(), "KIT": kit, "json": json, "os": os, "time": time,
     "RERUN": True, "MODE": "submit", "__name__": "contract"}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b32m_contract.py")).read(), g)
res = g["contract_run"](base, wait_s=1800)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
