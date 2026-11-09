#!/usr/bin/env python3
"""E062: a replica run cut at solver hour H, as a flat "live mirror" that replica_report.py reads.

  cut_run.py RUN_DIR H OUT_DIR
RUN_DIR is a pod run dir (out/dsv41_logs/...) or a flat mirror. Keeps attempts with t_end_h <= H, metrics windows
with elapsed_h <= H, solver.log lines stamped "+<h>h" <= H, and the server logs as they are (their 10-s rows are
mapped to solver time by the report; rows after H only describe throughput, not the score). env.txt and
replica.json are copied so the report finds the workload. The report then rebuilds the submission from the records
the way the solver ranks answers, so two runs cut at the same H compare like-for-like (`--baseline`).
"""

import json
import re
import shutil
import sys
from pathlib import Path

run, H, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
logs = run / "out" / "dsv41_logs" if (run / "out" / "dsv41_logs").is_dir() else run
out.mkdir(parents=True, exist_ok=True)
n = k = 0
with open(logs / "attempts.jsonl") as f, open(out / "attempts.jsonl", "w") as g:
    for line in f:
        n += 1
        r = json.loads(line)
        if (r.get("t_end_h") or 0) <= H:
            g.write(line)
            k += 1
if (logs / "metrics.jsonl").exists():
    with open(logs / "metrics.jsonl") as f, open(out / "metrics.jsonl", "w") as g:
        g.writelines(line for line in f if (json.loads(line).get("elapsed_h") or 0) <= H)
stamp = re.compile(r"^\[\S+ \+([0-9.]+)h\]")
with open(logs / "solver.log") as f, open(out / "solver.log", "w") as g:
    for line in f:
        m = stamp.match(line)
        if m and float(m.group(1)) > H:
            break
        g.write(line)
for name in ("status.json", "server.log", "server_cmd.json"):
    if (logs / name).exists():
        shutil.copy(logs / name, out / name)
for name in ("env.txt", "replica.json"):
    if (run / name).exists():
        shutil.copy(run / name, out / name)
print(f"{run}: kept {k}/{n} attempt records with t_end_h <= {H} -> {out}")
