"""Hour-by-hour serving metrics of a replica next to V12 text at the same Kaggle hour (for "V12 + images").
Per hour window [h-1, h):
- from the solver's metrics.jsonl (5-min windows, weighted by window_s): prefix-cache hit rate, uncached prompt tok/s,
  preemptions (sum), server running, solver in-flight (running), host-KV restored tok/s, MM processor-cache hit rate
  (image run only: mm_hit / mm_queries);
- from attempts finished in the hour: uncached prompt tokens per turn = sum(prompt - cache_hit) / sum(turns), and
  prompt tokens per turn;
- from the sampler's last sample of the hour: API-server + EngineCore RSS (where the MM processor caches live) and the
  worst cap - reserved;
- score so far over the workload and tasks answered at the hour's end.
usage: hourly_extra_vs_v12.py SOL NEW_METRICS NEW_ATTEMPTS NEW_SAMP NEW_T0 V12_METRICS V12_ATTEMPTS V12_SAMP V12_T0 NOW_H"""
import json, sys
sol = json.load(open(sys.argv[1]))
def depth(x):
    d = 0
    while isinstance(x, list) and x: x = x[0]; d += 1
    return d
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def run(mp, ap, sp, t0):
    return {"m": load(mp), "a": load(ap), "s": load(sp), "t0": float(t0)}
def window(R, h0, h1):
    ms = [m for m in R["m"] if h0 <= m.get("elapsed_h", -1) < h1 and m.get("window_s", 0) > 100]
    w = sum(m["window_s"] for m in ms) or 1
    avg = lambda k: sum(m.get(k, 0) * m["window_s"] for m in ms) / w if ms else None
    mq = sum(m.get("mm_queries", 0) or 0 for m in ms); mh = [m.get("mm_hit") for m in ms if m.get("mm_hit") is not None]
    at = [a for a in R["a"] if h0 <= float(a.get("t_end_h") or 99) < h1 and a.get("usage")]
    turns = sum(a.get("turns") or 0 for a in at) or 0
    unc = sum((a["usage"].get("prompt", 0) - a["usage"].get("cache_hit", 0)) for a in at)
    pr = sum(a["usage"].get("prompt", 0) for a in at)
    last = None
    for s in R["s"]:
        if (s["t"] - R["t0"]) / 3600 > h1: break
        last = s
    rss = cr = None
    if last:
        g = last.get("rss_gib") or {}; rss = (g.get("api_server") or 0) + (g.get("engine_core") or 0)
        c = last.get("alloc_cap") or {}
        v = [x["cap_minus_reserved_now"] for x in c.values() if x.get("cap_minus_reserved_now") is not None]
        cr = min(v) if v else None
    best = {}
    for a in R["a"]:
        if float(a.get("t_end_h") or 99) > h1 or a["task_id"] not in sol or a.get("pred") is None: continue
        t = a["task_id"]; n = len(sol[t]); p = a["pred"]; preds = p if (n > 1 and depth(p) == 3) else [p]
        ok = [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
        best[t] = [x or y for x, y in zip(best.get(t, [False] * n), ok)]
    score = 100 * sum(sum(v) / len(v) for v in best.values()) / len(sol)
    return {"hit": avg("prefix_hit"), "unc_s": avg("uncached_tok_s"), "pre": sum(m.get("preemptions", 0) for m in ms) if ms else None,
            "srv": avg("srv_running"), "inflight": avg("running"), "host": avg("host_tok_s"), "gen": avg("gen_tok_s"),
            "mm": (sum(mh) / len(mh)) if mh else None, "mmq": mq, "unc_turn": unc / turns if turns else None,
            "pr_turn": pr / turns if turns else None, "rss": rss, "cr": cr, "score": score, "ans": len(best)}
args = sys.argv[2:]
new, old = run(*args[0:4]), run(*args[4:8]); now = float(args[8])
f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
print("| Kaggle h | score (new / V12) | tasks answered | gen tok/s | prefix hit | uncached tok/s | uncached prompt tok/turn | prompt tok/turn | preemptions | server running | solver in-flight | host-KV restore tok/s | MM cache hit (new) | API+Engine RSS GiB | worst cap-reserved |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
h = 1 if now >= 1 else now
while h <= now + 1e-9:
    a, b = window(new, h - 1 if h >= 1 else 0, h), window(old, h - 1 if h >= 1 else 0, h)
    pair = lambda k, d=1: f"{f(a[k], d)} / {f(b[k], d)}"
    print(f"| {h:g} | {pair('score')} | {a['ans']} / {b['ans']} | {pair('gen', 0)} | {pair('hit', 3)} | {pair('unc_s', 0)} | {pair('unc_turn', 0)} | {pair('pr_turn', 0)} | {pair('pre', 0)} | {pair('srv')} | {pair('inflight')} | {pair('host', 0)} | {f(a['mm'], 3)} ({a['mmq']:.0f} q) | {pair('rss', 2)} | {pair('cr', 2)} |")
    h = h + 1 if h + 1 <= now else (now if h < now else h + 1)
