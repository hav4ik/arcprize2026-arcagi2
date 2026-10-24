#!/usr/bin/env python3
"""E043: parse vLLM server logs into time series (for throughput / cache-health figures in the tech report).

  server_log_series.py --runs runs/tp4A runs/tp4B ... --out server_series.json
Per run: the 10 s engine stats lines [date, time, prompt tok/s, generation tok/s, running, waiting, KV usage %,
prefix-cache hit %], DSpark lines [date, time, mean acceptance length, accepted tok/s, drafted tok/s] when present,
and the E043 control-file loads (the runtime mask in effect) with the line number of the stats series they precede.
"""
import argparse
import json
import re
from pathlib import Path

STATS = re.compile(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine \d+: Avg prompt throughput: ([\d.]+) tokens/s, "
                   r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                   r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
SPEC = re.compile(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d) \[metrics\.py:\d+\] SpecDecoding metrics: Mean acceptance length: ([\d.]+), "
                  r"Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: ([\d.]+) tokens/s")
CTRL = re.compile(r"Worker_TP0\S* pid=\d+\) \[E043\] ctrl loaded: experts=(\S+) engram=(\S+) observe=(\S+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for r in a.runs:
        stats, spec, ctrl = [], [], []
        log = Path(r) / "server.log"
        if not log.exists():
            continue
        for line in open(log, errors="replace"):
            m = STATS.search(line)
            if m:
                d, t, *v = m.groups()
                stats.append([d, t, *map(float, v)])
                continue
            m = SPEC.search(line)
            if m:
                d, t, *v = m.groups()
                spec.append([d, t, *map(float, v)])
                continue
            m = CTRL.search(line)
            if m:
                ctrl.append({"at_stats_index": len(stats), "experts": m.group(1), "engram": m.group(2)})
        cmd = (Path(r) / "cmd.txt").read_text().strip() if (Path(r) / "cmd.txt").exists() else ""
        out[Path(r).name] = {"cmd": cmd, "columns": ["date", "time", "prompt_tps", "gen_tps", "running", "waiting",
                                                     "kv_usage_pct", "prefix_hit_pct"],
                             "stats": stats, "spec_columns": ["date", "time", "mean_accept_len", "accepted_tps", "drafted_tps"],
                             "spec": spec, "ctrl_loads": ctrl}
        print(Path(r).name, len(stats), "stats lines,", len(spec), "spec lines,", len(ctrl), "ctrl loads")
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
