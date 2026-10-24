#!/usr/bin/env python3
"""E042: per-rank GPU kernel totals from vLLM torch-profiler traces (one *.json.gz per rank).

Usage: prof_summary.py <trace dir> [--json out.json]. For each rank: span, busy time, kernel count, time per
category (all-reduce, Marlin MoE, indexer, dense MXFP8, attention, Engram/all-gather, other) and the top kernels.
Per-step figures need the step count, which is taken as the number of `_mxfp4_indexer_logits_kernel` launches
divided by `--indexer-per-step` (default 0 = not computed).
"""

import argparse
import collections
import glob
import gzip
import json
import re

CATS = [("allreduce", r"AllReduce|allreduce|all_reduce|cross_device"),
        ("allgather", r"AllGather|all_gather"),
        ("marlin_moe", r"marlin"),
        ("indexer", r"indexer"),
        ("dense_mxfp8", r"mxfp8|_ada_"),
        ("attention", r"deepseek_v41|sparse_attention|prefill_kernel|flash|attn"),
        ("other", r".")]


def summarize(path, top):
    ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        return {"file": path, "kernels": 0}
    by, names, counts = collections.Counter(), collections.Counter(), collections.Counter()
    for e in k:
        n = e.get("name", "")
        names[n[:100]] += e["dur"]
        counts[n[:100]] += 1
        for c, pat in CATS:
            if re.search(pat, n):
                by[c] += e["dur"]
                break
    t0 = min(e["ts"] for e in k)
    t1 = max(e["ts"] + e["dur"] for e in k)
    return {"file": path.split("/")[-1], "span_ms": round((t1 - t0) / 1e3, 1), "busy_ms": round(sum(by.values()) / 1e3, 1),
            "kernels": len(k), "category_ms": {c: round(v / 1e3, 1) for c, v in by.most_common()},
            "indexer_launches": sum(v for n, v in counts.items() if "_mxfp4_indexer_logits_kernel" in n),
            "top": [{"name": n, "ms": round(v / 1e3, 2), "count": counts[n]} for n, v in names.most_common(top)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir")
    p.add_argument("--json")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--indexer-per-step", type=int, default=0)
    a = p.parse_args()
    files = sorted(glob.glob(a.trace_dir + "/**/*.json*", recursive=True))
    out = []
    for f in files:
        r = summarize(f, a.top)
        if a.indexer_per_step and r.get("indexer_launches"):
            steps = r["indexer_launches"] / a.indexer_per_step
            r["steps_est"] = round(steps, 1)
            r["per_step_ms"] = {c: round(v / steps, 2) for c, v in r["category_ms"].items()}
        out.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "top"}), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
