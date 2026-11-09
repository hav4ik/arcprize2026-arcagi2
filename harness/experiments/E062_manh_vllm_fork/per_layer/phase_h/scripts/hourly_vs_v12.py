"""Hour-by-hour comparison of a replica with the V12 replica at the same Kaggle hour (clock = hours since the
notebook start): ARC score so far over the workload (acc_series method: best of finished attempts, unanswered = 0),
first / second attempt accuracy so far (fully correct / finished attempts with an answer), mean gen tok/s and running
over the hour, worst cap - reserved and Kaggle-eq RAM (cgroup form) at the hour's end.
usage: hourly_vs_v12.py SOL.json NEW_ATTEMPTS NEW_TPS.tsv NEW_SAMP.jsonl NEW_T0 V12_ATTEMPTS V12_TPS.tsv V12_SAMP.jsonl V12_T0 [now_h]"""
import calendar, json, sys, time
sol = json.load(open(sys.argv[1]))
def depth(x):
    d = 0
    while isinstance(x, list) and x:
        x = x[0]; d += 1
    return d
def pairs_ok(a):
    t = a["task_id"]; n = len(sol[t]); p = a.get("pred")
    if p is None: return None
    preds = p if (n > 1 and depth(p) == 3) else [p]
    return [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
def acc_at(path, h):
    best, acc = {}, {0: [0, 0], 1: [0, 0]}
    for l in open(path):
        if not l.strip(): continue
        a = json.loads(l)
        if float(a.get("t_end_h") or 99) > h or a["task_id"] not in sol: continue
        ok = pairs_ok(a)
        if ok is None: continue
        t = a["task_id"]; best[t] = [x or y for x, y in zip(best.get(t, [False] * len(ok)), ok)]
        k = int(a.get("attempt", 0))
        if k in acc: acc[k][0] += 1; acc[k][1] += all(ok)
    sc = 100 * sum(sum(v) / len(v) for v in best.values()) / len(sol)
    return sc, acc
def tps_at(path, h0, h1):
    xs = []
    for l in open(path):
        p = l.rstrip("\n").split("\t")
        if p[0] == "utc" or len(p) < 5: continue
        try: kh, g, r = float(p[1]), float(p[2]), float(p[3])
        except ValueError: continue
        if h0 <= kh < h1 and r > 0: xs.append((g, r))
    return (sum(g for g, _ in xs) / len(xs), sum(r for _, r in xs) / len(xs)) if xs else (None, None)
def samp_at(path, t0, h):
    last = None
    for l in open(path):
        r = json.loads(l)
        if (r["t"] - t0) / 3600 > h: break
        last = r
    if not last: return None, None
    c = last.get("alloc_cap") or {}
    cr = min([v["cap_minus_reserved_now"] for v in c.values() if v.get("cap_minus_reserved_now") is not None] or [None], key=lambda x: 99 if x is None else x)
    keq = last.get("kaggle_eq_gib") if last.get("cgroup_gib") else None
    return cr, keq
args = sys.argv[2:]
new = dict(zip(("att", "tps", "samp", "t0"), args[0:4])); old = dict(zip(("att", "tps", "samp", "t0"), args[4:8]))
now_h = float(args[8]) if len(args) > 8 else 99
f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
print("| Kaggle h | score (new / V12) | 1st-att acc so far (new / V12) | 2nd-att acc so far (new / V12) | gen tok/s in hour (new / V12) | running (new / V12) | worst cap-reserved GiB (new / V12) | Kaggle-eq GiB (new / V12) |")
print("|---|---|---|---|---|---|---|---|")
h = 1 if now_h >= 1 else now_h
while h <= min(now_h, 12) + 1e-9:
    sn, an = acc_at(new["att"], h); so, ao = acc_at(old["att"], h)
    pa = lambda a, k: f"{a[k][1]}/{a[k][0]} = {100 * a[k][1] / a[k][0]:.1f}%" if a[k][0] else "-"
    gn, rn = tps_at(new["tps"], h - 1, h); go, ro = tps_at(old["tps"], h - 1, h)
    cn, kn = samp_at(new["samp"], float(new["t0"]), h); co, ko = samp_at(old["samp"], float(old["t0"]), h)
    print(f"| {h:g} | {sn:.1f} / {so:.1f} | {pa(an, 0)} / {pa(ao, 0)} | {pa(an, 1)} / {pa(ao, 1)} | {f(gn, 0)} / {f(go, 0)} | {f(rn)} / {f(ro)} | {f(cn, 2)} / {f(co, 2)} | {f(kn)} / {f(ko)} |")
    h = h + 1 if h + 1 <= now_h or h >= now_h else now_h
