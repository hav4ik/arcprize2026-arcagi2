"""Freeze a run's current series into frozen_<slug>.json (mem, tps, acc, outages, runs_row) so the builder never rebuilds it.
usage: freeze_run.py "<run name>" <slug> "<status text>" [elapsed_h]   (run update_dash.py first so the files are current)"""
import json, os, sys
D = os.environ.get("DASH_DIR", "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/dashboard")
name, slug, status = sys.argv[1:4]
new = os.environ.get("NEWNAME", name)  # the frozen copy may carry a new run name
f = {k: [x for x in json.load(open(f"{D}/{k}.json")) if x["run"] == name] for k in ("mem", "tps", "acc", "outages")}
row = [r for r in json.load(open(f"{D}/runs.json")) if r["run"] == name]
assert row, "no runs row"
row[0]["status"] = status
if len(sys.argv) > 4: row[0]["elapsed_h"] = float(sys.argv[4])
f["runs_row"] = row
for k, v in f.items():
    for x in v:
        if x.get("run") == name: x["run"] = new
json.dump(f, open(f"{D}/frozen_{slug}.json", "w"), indent=1)
print({k: len(v) for k, v in f.items()})
