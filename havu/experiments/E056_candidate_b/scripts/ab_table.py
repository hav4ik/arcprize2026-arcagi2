#!/usr/bin/env python3
"""E056: the late-phase A/B table (offload vs candB) from the copied results.

Reads results/runs/<TAG>.progress (replay start lines, E051 run_capacity.sh format) and, per arm dir
results/runs/<TAG>_<arm>/: replay_C*_D*.json (E051 tool_replay_kv.py) and metrics.jsonl (E051 metrics_logger.py, 5 s).
The measurement window is the replay's own (start + setup + drain + 600 s warm-up, window_s long), as E051's
summarize_capacity.py computes it.

    python3 scripts/ab_table.py results/runs late [--md]
"""
import datetime as dt
import glob
import json
import os
import re
import sys

ROOT, TAG = sys.argv[1], sys.argv[2]
WARM = 600


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


starts = {}
for line in open(os.path.join(ROOT, f"{TAG}.progress")):
    m = re.match(r"(\S+) (\S+): replay C=(\d+) .* start$", line.strip())
    if m:
        starts[m[2]] = ts(m[1])

rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, f"{TAG}_*", "replay_C*_D*.json"))):
    arm_dir = os.path.basename(os.path.dirname(f))
    r = json.load(open(f))
    t0 = starts[arm_dir]
    w0 = t0 + r["setup_s"] + r["drain_s"] + WARM
    met = [json.loads(l) for l in open(os.path.join(os.path.dirname(f), "metrics.jsonl"))]
    met = [m for m in met if "vllm:kv_cache_usage_perc" in m and w0 <= m["t"] <= w0 + r["window_s"]]
    kv = [m["vllm:kv_cache_usage_perc"] for m in met]
    q = r["prompt_tokens_queried"]
    gpu_hit_tok = q - r["prompt_tokens_uncached"]
    rows[arm_dir.removeprefix(TAG + "_")] = {
        "gen tok/s": r["gen_tok_s"],
        "gen tok/s (server counter)": r["gen_tok_s_metrics"],
        "attempts / h": round(3600 * r["attempts_done_in_window"] / r["window_s"], 1),
        "turns / h": round(3600 * r["turns_done_in_window"] / r["window_s"], 1),
        "attempts / turns in window": f'{r["attempts_done_in_window"]} / {r["turns_done_in_window"]}',
        "prompt tokens queried (window)": q,
        "GPU prefix hit": r["hit_rate"],
        "host-restored tokens (external hits)": r["external_hit_tokens"],
        "host-restored share of queried": round(r["external_hit_tokens"] / q, 3) if q else None,
        "true uncached share": round(r["prompt_tokens_uncached_true"] / q, 3) if q else None,
        "true uncached tok/s": r["uncached_true_tok_s"],
        "turn uncached mean / p99": f'{r["turn_uncached_mean"]} / {r["turn_uncached_p99"]}',
        "turns > 20k uncached": r["turns_uncached_gt20k"],
        "full re-prefills (>= 50k, <= 640 cached)": r["full_reprefills_50k"],
        "preemptions": r["preemptions"],
        "TTFT p50 / p99 (s)": f'{r["ttft_median_s"]} / {r["ttft_p99_s"]}',
        "turn p50 / p99 (s)": f'{r["turn_s_median"]} / {r["turn_s_p99"]}',
        "KV mean / peak (window)": (f"{sum(kv) / len(kv):.3f} / {max(kv):.3f}" if kv else None),
        "running mean / waiting mean (max)": (
            f'{sum(m.get("vllm:num_requests_running", 0) for m in met) / len(met):.1f} / '
            f'{sum(m.get("vllm:num_requests_waiting", 0) for m in met) / len(met):.1f} '
            f'({max(m.get("vllm:num_requests_waiting", 0) for m in met):.0f})' if met else None),
        "host load / store MB/s": f'{r["load_MB_s"]} / {r["store_MB_s"]}',
        "loads (ms, MB each)": f'{r["loads"]} ({r["ms_per_load"]}, {r["MB_per_load"]})',
        "server PSS max (GiB)": max((m.get("server_pss_GiB", 0) for m in met), default=None),
        "Shmem max (GiB)": max((m.get("Shmem_GiB", 0) for m in met), default=None),
        "setup / drain / window (s)": f'{r["setup_s"]} / {r["drain_s"]} / {r["window_s"]}',
        "slots started at setup": r["slots_started_at_setup"],
        "turn prompt p50 / p90": "/".join(map(str, r["turn_prompt_p50_p90"] or [])),
    }

arms = sorted(rows, key=lambda a: (a != "offload", a))
if "--md" in sys.argv:
    print("| | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for k in rows[arms[0]]:
        print(f"| {k} | " + " | ".join(str(rows[a][k]) for a in arms) + " |")
else:
    print(json.dumps(rows, indent=1))
