#!/usr/bin/env python3
"""Summarize an E051 capacity matrix (run_capacity.sh): one row per replay (arm x streams x tool idle).

Reads /root/e051/runs/<TAG>_<arm>/replay_C*_D*.json and the server metrics log (metrics.jsonl, 5 s): KV usage,
running/waiting requests and host memory in the replay's measurement window (start line in <TAG>.progress + setup +
drain + 600 s warm-up), plus the KV peak over the whole replay (kv_peak_run, includes the cold setup).

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
WARM = 600


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


spans = {}
for line in open(os.path.join(ROOT, f"{TAG}.progress")):
    m = re.match(r"(\S+) (\S+): replay C=(\d+) D=(\S+?)(?: S=(\S+))?(?: T=(\S+))?(?: M=(\S+))? (start|rc=)", line)
    if m:
        key = (m[2], int(m[3]), m[4], float(m[5] or 1.0), float(m[6] or 1.0), int(m[7] or 0))
        spans.setdefault(key, [None, None])[0 if m[8] == "start" else 1] = ts(m[1])

COLS = ["arm", "C", "D", "S", "T", "M", "gen_tok_s", "hit_gpu", "ext_hit_tok", "uncached_true_tok_s", "turn_unc_mean", "turn_unc_p99",
        "unc>20k", "reprefill50k", "preempt", "load_MB_s", "store_MB_s", "ms_load", "MB_load", "ttft_p50", "ttft_p99",
        "turn_p50", "turn_p99", "turns", "attempts", "att_per_h", "turns_per_h", "kv_peak", "kv_mean", "kv_peak_run", "running_mean",
        "waiting_mean", "waiting_max", "shmem_GiB", "pss_GiB"]
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, f"{TAG}_*", "replay_C*_D*.json"))):
    arm = os.path.basename(os.path.dirname(f))
    r = json.load(open(f))
    C, D = r["conc"], r["tool_delay_s"]
    D = int(D) if float(D).is_integer() else D
    S = float(r.get("scale", 1.0))
    T, M = float(r.get("tail_frac", 1.0)), int(r.get("min_start_ctx", 0))
    t0, t1 = spans.get((arm, C, str(D), S, T, M), [None, None])
    allm = [json.loads(l) for l in open(os.path.join(os.path.dirname(f), "metrics.jsonl"))]
    allm = [m for m in allm if "vllm:kv_cache_usage_perc" in m]
    # measurement window: replay start + setup + drain + warm-up (WARM s, run_capacity.sh default 600), for window_s
    w0 = t0 + r["setup_s"] + r["drain_s"] + WARM if t0 else None
    met = [m for m in allm if w0 and w0 <= m["t"] <= w0 + r["window_s"]]
    run = [m for m in allm if t0 and t1 and t0 <= m["t"] <= t1]
    kv = [m["vllm:kv_cache_usage_perc"] for m in met]
    rows.append({
        "arm": arm.removeprefix(TAG + "_"), "C": C, "D": D, "S": S, "T": T, "M": M, "gen_tok_s": r["gen_tok_s"], "hit_gpu": r["hit_rate"],
        "ext_hit_tok": r.get("external_hit_tokens"), "uncached_true_tok_s": r.get("uncached_true_tok_s"),
        "turn_unc_mean": r.get("turn_uncached_mean"), "turn_unc_p99": r.get("turn_uncached_p99"),
        "unc>20k": r.get("turns_uncached_gt20k"), "reprefill50k": r.get("full_reprefills_50k"),
        "preempt": r["preemptions"], "load_MB_s": r.get("load_MB_s"), "store_MB_s": r.get("store_MB_s"),
        "ms_load": r.get("ms_per_load"), "MB_load": r.get("MB_per_load"), "ttft_p50": r["ttft_median_s"],
        "ttft_p99": r.get("ttft_p99_s"), "turn_p50": r["turn_s_median"], "turn_p99": r.get("turn_s_p99"),
        "turns": r["turns_done_in_window"], "attempts": r["attempts_done_in_window"],
        "att_per_h": round(3600 * r["attempts_done_in_window"] / r["window_s"], 1),
        "turns_per_h": round(3600 * r["turns_done_in_window"] / r["window_s"], 1),
        "kv_peak": round(max(kv), 3) if kv else None, "kv_mean": round(sum(kv) / len(kv), 3) if kv else None,
        "kv_peak_run": round(max(m["vllm:kv_cache_usage_perc"] for m in run), 3) if run else None,
        "running_mean": round(sum(m.get("vllm:num_requests_running", 0) for m in met) / len(met), 1) if met else None,
        "waiting_mean": round(sum(m.get("vllm:num_requests_waiting", 0) for m in met) / len(met), 1) if met else None,
        "waiting_max": max((m.get("vllm:num_requests_waiting", 0) for m in met), default=None),
        "shmem_GiB": max((m.get("Shmem_GiB", 0) for m in met), default=None),
        "pss_GiB": max((m.get("server_pss_GiB", 0) for m in met), default=None),
    })
rows.sort(key=lambda x: (x["S"], x["T"], x["M"], x["C"], x["D"], x["arm"]))
if MD:
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for x in rows:
        print("| " + " | ".join(str(x[c]) for c in COLS) + " |")
else:
    for x in rows:
        print(json.dumps(x))
