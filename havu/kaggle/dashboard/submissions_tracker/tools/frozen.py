"""Finished runs come from frozen_*.json files in the dashboard dir (keys mem, tps, acc, runs_row, outages) and are never rebuilt."""
import glob, json, os
D = os.environ.get("DASH_DIR", "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/dashboard")
def load():
    out = {}
    for p in sorted(glob.glob(f"{D}/frozen_*.json")):
        f = json.load(open(p))
        row = (f.get("runs_row") or [None])[0]
        if row:
            out[row["run"]] = f
    return out
