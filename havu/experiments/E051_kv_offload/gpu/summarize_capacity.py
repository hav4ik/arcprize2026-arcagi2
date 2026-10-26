#!/usr/bin/env python3
"""Summarize an E051 capacity matrix (run_capacity.sh): one row per replay (arm x streams x tool idle).

Reads /root/e051/runs/<TAG>_<arm>/replay_C*_D*.json, the server metrics log (metrics.jsonl, 5 s) for peak KV usage,
running/waiting requests and host memory between the replay's start and end lines in <TAG>.progress.

    python summarize_capacity.py /root/e051/runs cap2 [--md]
"""
import datetime as dt
import glob
import json
import os
import re
import sys

ROOT, TAG = sys.argv[1], sys.argv[2]
MD = "--md" in sys.argv


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


spans = {}
for line in open(os.path.join(ROOT, f"{TAG}.progress")):
    m = re.match(r"(\S+) (\S+): replay C=(\d+) D=(\S+) (start|rc=)", line)
    if m:
        key = (m[2], int(m[3]), m[4])
        spans.setdefault(key, [None, None])[0 if m[5] == "start" else 1] = ts(m[1])

COLS = ["arm", "C", "D", "gen_tok_s", "hit_gpu", "ext_hit_tok", "uncached_true_tok_s", "turn_unc_mean", "turn_unc_p99",
        "unc>20k", "reprefill50k", "preempt", "load_MB_s", "store_MB_s", "ms_load", "MB_load", "ttft_p50", "ttft_p99",
        "turn_p50", "turn_p99", "turns", "attempts", "kv_peak", "kv_mean", "waiting_max", "shmem_GiB", "pss_GiB"]
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, f"{TAG}_*", "replay_C*_D*.json"))):
    arm = os.path.basename(os.path.dirname(f))
    r = json.load(open(f))
    C, D = r["conc"], r["tool_delay_s"]
    D = int(D) if float(D).is_integer() else D
    t0, t1 = spans.get((arm, C, str(D)), [None, None])
    met = [json.loads(l) for l in open(os.path.join(os.path.dirname(f), "metrics.jsonl"))]
    met = [m for m in met if t0 and t1 and t0 <= m["t"] <= t1 and "vllm:kv_cache_usage_perc" in m]
    kv = [m["vllm:kv_cache_usage_perc"] for m in met]
    rows.append({
        "arm": arm.removeprefix(TAG + "_"), "C": C, "D": D, "gen_tok_s": r["gen_tok_s"], "hit_gpu": r["hit_rate"],
        "ext_hit_tok": r.get("external_hit_tokens"), "uncached_true_tok_s": r.get("uncached_true_tok_s"),
        "turn_unc_mean": r.get("turn_uncached_mean"), "turn_unc_p99": r.get("turn_uncached_p99"),
        "unc>20k": r.get("turns_uncached_gt20k"), "reprefill50k": r.get("full_reprefills_50k"),
        "preempt": r["preemptions"], "load_MB_s": r.get("load_MB_s"), "store_MB_s": r.get("store_MB_s"),
        "ms_load": r.get("ms_per_load"), "MB_load": r.get("MB_per_load"), "ttft_p50": r["ttft_median_s"],
        "ttft_p99": r.get("ttft_p99_s"), "turn_p50": r["turn_s_median"], "turn_p99": r.get("turn_s_p99"),
        "turns": r["turns_done_in_window"], "attempts": r["attempts_done_in_window"],
        "kv_peak": round(max(kv), 3) if kv else None, "kv_mean": round(sum(kv) / len(kv), 3) if kv else None,
        "waiting_max": max((m.get("vllm:num_requests_waiting", 0) for m in met), default=None),
        "shmem_GiB": max((m.get("Shmem_GiB", 0) for m in met), default=None),
        "pss_GiB": max((m.get("server_pss_GiB", 0) for m in met), default=None),
    })
rows.sort(key=lambda x: (x["C"], x["D"], x["arm"]))
if MD:
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for x in rows:
        print("| " + " | ".join(str(x[c]) for c in COLS) + " |")
else:
    for x in rows:
        print(json.dumps(x))
