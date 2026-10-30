#!/usr/bin/env python3
"""Summarise E067 kvcache runs: one row per run, plus optional time-binned series.

Usage: analyze.py RUN [RUN ...] [--window 60 300] [--series BIN_S] [--md]
RUN is a prefix in runs/ (e.g. e1_c32) or a path prefix.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(prefix):
    if not os.path.exists(prefix + ".turns.jsonl"):
        prefix = os.path.join(HERE, "runs", prefix)
    turns, attempts = [], []
    for line in open(prefix + ".turns.jsonl"):
        r = json.loads(line)
        (turns if "turn" in r else attempts).append(r)
    mets = [json.loads(l) for l in open(prefix + ".metrics.jsonl")]
    cfg = json.load(open(prefix + ".config.json"))
    srv = json.load(open(prefix + ".server.json")) if os.path.exists(prefix + ".server.json") else {}
    return prefix, turns, attempts, mets, cfg, srv


def at(mets, t):
    """metrics sample closest to time t (with real metrics)."""
    good = [m for m in mets if "vllm:generation_tokens_total" in m]
    return min(good, key=lambda m: abs(m["t"] - t))


def d(m0, m1, k):
    return m1.get(k, 0.0) - m0.get(k, 0.0)


def ideal_reuse(turns, w0, w1):
    """(ideal, client) cache fractions for turns sent in [w0, w1]. ideal = everything except the
    newest tool output was seen before; client = usage.prompt_tokens_details.cached_tokens."""
    by = {}
    for r in sorted(turns, key=lambda r: (r["aid"], r["turn"])):
        by.setdefault(r["aid"], []).append(r)
    tot = reuse = cached = 0
    for rows in by.values():
        prev = None
        for r in rows:
            if w0 <= r["t_send"] <= w1 and not r["err"]:
                tot += r["prompt"]
                reuse += 400 if prev is None else prev["prompt"] + prev["out"]
                cached += r["cached"] or 0
            prev = r
    return reuse / max(tot, 1), cached / max(tot, 1)


def summarize(prefix, w0, w1):
    prefix, turns, attempts, mets, cfg, srv = load(prefix)
    w1 = min(w1, cfg["duration"])
    m0, m1 = at(mets, w0), at(mets, w1)
    dt = m1["t"] - m0["t"]
    gen = d(m0, m1, "vllm:generation_tokens_total")
    computed = d(m0, m1, "vllm:iteration_tokens_total_sum") - gen
    ptoks = d(m0, m1, "vllm:prompt_tokens_total")
    lc = d(m0, m1, "vllm:prompt_tokens_by_source_total{local_compute}")
    lh = d(m0, m1, "vllm:prompt_tokens_by_source_total{local_cache_hit}")
    ex = d(m0, m1, "vllm:prompt_tokens_by_source_total{external_kv_transfer}")
    pq = d(m0, m1, "vllm:prefix_cache_queries_total")
    ph = d(m0, m1, "vllm:prefix_cache_hits_total")
    eq = d(m0, m1, "vllm:external_prefix_cache_queries_total")
    eh = d(m0, m1, "vllm:external_prefix_cache_hits_total")
    pre = d(m0, m1, "vllm:num_preemptions_total")
    win = [m for m in mets if w0 <= m["t"] <= w1 and "vllm:num_requests_running" in m]
    run = np.array([m["vllm:num_requests_running"] for m in win])
    wait = np.array([m["vllm:num_requests_waiting"] for m in win])
    kv = np.array([m["vllm:kv_cache_usage_perc"] for m in win])
    rss = [m["server_rss_mb"] for m in mets if m.get("server_rss_mb")]
    tw = [r for r in turns if w0 <= r["t_send"] <= w1 and not r["err"]]
    tdone = [r for r in turns if not r["err"] and w0 <= r["t_send"] + r["lat"] <= w1]
    ttft = np.array([r["ttft"] for r in tw if r["ttft"] is not None])
    lat = np.array([r["lat"] for r in tw])
    # censored: requests still in flight at the end (no record) can't be seen; errors are
    errs = [r for r in turns if r["err"]]
    off_g2c = d(m0, m1, "vllm:kv_offload_total_bytes_total{GPU_to_CPU}")
    off_c2g = d(m0, m1, "vllm:kv_offload_total_bytes_total{CPU_to_GPU}")
    pct = (lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan"))
    return {
        "run": os.path.basename(prefix), "C": cfg["concurrency"],
        "gate": cfg["gate_kv"] if cfg["gate_kv"] < 1 else "-",
        "kv_tok": srv.get("kv_tokens"),
        "gen_tok_s": gen / dt, "turns_min": len(tdone) / dt * 60,
        "att_done": sum(1 for a in attempts if a["attempt_done"]),
        "prompt_tok_s": ptoks / dt, "computed_tok_s": computed / dt,
        "recompute_frac": (computed - lc) / max(computed, 1),
        "hit_ideal": ideal_reuse(turns, w0, w1)[0], "hit_client": ideal_reuse(turns, w0, w1)[1],
        "hit_gpu_req": lh / max(ptoks, 1), "hit_ext_req": ex / max(ptoks, 1),
        "hit_gpu_prom": ph / max(pq, 1), "prom_q_over_prompt": pq / max(ptoks, 1),
        "hit_ext_prom": eh / max(eq, 1) if eq else float("nan"),
        "preempt_min": pre / dt * 60,
        "running": run.mean() if len(run) else float("nan"), "waiting": wait.mean() if len(wait) else float("nan"),
        "wait_max": wait.max() if len(wait) else float("nan"), "kv_use": kv.mean() if len(kv) else float("nan"),
        "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99),
        "ttft_max": ttft.max() if len(ttft) else float("nan"),
        "lat_p50": pct(lat, 50), "lat_p99": pct(lat, 99),
        "n_ttft_gt10": int((ttft > 10).sum()), "n_ttft_gt30": int((ttft > 30).sum()), "n_req": len(tw),
        "errors": len(errs), "rss_max_gb": max(rss) / 1024 if rss else float("nan"),
        "g2c_GB": off_g2c / 1e9, "c2g_GB": off_c2g / 1e9,
        "gate_wait_p50": pct(np.array([r["gate_wait"] for r in turns if r.get("gate_wait") is not None]), 50),
    }


def series(prefix, bin_s):
    prefix, turns, attempts, mets, cfg, srv = load(prefix)
    good = [m for m in mets if "vllm:generation_tokens_total" in m]
    rows = []
    t_end = good[-1]["t"]
    t = 0.0
    while t + bin_s <= t_end + 1e-6:
        a, b = at(good, t), at(good, t + bin_s)
        dt = max(b["t"] - a["t"], 1e-9)
        inb = [m for m in good if t <= m["t"] < t + bin_s]
        pt = d(a, b, "vllm:prompt_tokens_total")
        gen = d(a, b, "vllm:generation_tokens_total")
        tt = [r["ttft"] for r in turns if t <= r["t_send"] < t + bin_s and r["ttft"] is not None]
        rows.append({
            "t": f"{int(t)}-{int(t + bin_s)}",
            "active": np.mean([m["active"] for m in inb]),
            "run": np.mean([m["vllm:num_requests_running"] for m in inb]),
            "wait": np.mean([m["vllm:num_requests_waiting"] for m in inb]),
            "kv": np.mean([m["vllm:kv_cache_usage_perc"] for m in inb]),
            "gen/s": gen / dt,
            "gpu_hit": d(a, b, "vllm:prompt_tokens_by_source_total{local_cache_hit}") / max(pt, 1),
            "ext_hit": d(a, b, "vllm:prompt_tokens_by_source_total{external_kv_transfer}") / max(pt, 1),
            "prom_hit": d(a, b, "vllm:prefix_cache_hits_total") / max(d(a, b, "vllm:prefix_cache_queries_total"), 1),
            "preempt": d(a, b, "vllm:num_preemptions_total"),
            "ttft_p90": float(np.percentile(tt, 90)) if tt else float("nan"),
        })
        t += bin_s
    return rows


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return "-"
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"
    return str(v)


def table(rows, cols, md):
    if md:
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    else:
        w = [max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols]
        out = ["  ".join(c.rjust(x) for c, x in zip(cols, w))]
        out += ["  ".join(fmt(r[c]).rjust(x) for c, x in zip(cols, w)) for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", nargs=2, type=float, default=[60, 300])
    ap.add_argument("--series", type=float, default=0)
    ap.add_argument("--cols", default="")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    if a.series:
        for r in a.runs:
            rows = series(r, a.series)
            print(f"\n### {os.path.basename(r)} ({int(a.series)} s bins)\n")
            print(table(rows, list(rows[0].keys()), a.md))
        return
    rows = [summarize(r, *a.window) for r in a.runs]
    cols = a.cols.split(",") if a.cols else list(rows[0].keys())
    print(table(rows, cols, a.md))


if __name__ == "__main__":
    main()
