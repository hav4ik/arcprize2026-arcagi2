#!/usr/bin/env python3
"""V1 (pod 4 b32m-fix replica) vs Manh's V12 replica (pod 2), read-only from local copies, up to Kaggle h H:
(a) gen tok/s, running, waiting, KV per half-hour (tps.tsv); (b) per finished attempt: generated tokens, turns, context
at the last turn (prompt + completion of the last usage event in the trace), p50 / p90, split by first attempts and all;
(d) first-attempt accuracy on the tasks both runs have finished (attempt 0, every test pair right).
Usage: cmp_v12.py <V1 data> <V1 traces dir> <V12 tps.tsv> <V12 attempts.jsonl> <V12 traces dir> <solutions> <H>"""
import glob, gzip, json, os, statistics, sys
from collections import defaultdict
v1d, v1tr, v12tps, v12att, v12tr, solp, H = sys.argv[1:8]; H = float(H)
sol = json.load(open(solp))


def tps(path):
    b = defaultdict(list)
    for l in list(open(path))[1:]:
        r = l.rstrip("\n").split("\t")
        h = float(r[1])
        if 0 <= h < H:
            b[int(h * 2) / 2].append((float(r[2]), float(r[3]), float(r[4]), float(r[5]) if r[5] else None))
    return b


a1, a12 = tps(f"{v1d}/tps.tsv"), tps(v12tps)
print("(a) per half-hour: gen tok/s / running / waiting / KV%  (V1 | V12)")
print("| Kaggle h | V1 gen | V1 run | V1 wait | V1 KV | V12 gen | V12 run | V12 wait | V12 KV |")
print("|---|---|---|---|---|---|---|---|---|")
m = lambda v, i: sum(x[i] for x in v if x[i] is not None) / max(1, sum(1 for x in v if x[i] is not None))
for k in sorted(set(a1) | set(a12)):
    x, y = a1.get(k, []), a12.get(k, [])
    f = lambda v: f"{m(v,0):.0f} | {m(v,1):.1f} | {m(v,2):.1f} | {m(v,3):.0f}" if v else "- | - | - | -"
    print(f"| {k:.1f}-{k+0.5:.1f} | {f(x)} | {f(y)} |")
tot = lambda b: (sum(m(v, 0) * len(v) for v in b.values()) / sum(len(v) for v in b.values()),
                 sum(m(v, 1) * len(v) for v in b.values()) / sum(len(v) for v in b.values()))
print(f"mean over 0-{H} h: V1 {tot(a1)[0]:.0f} tok/s at {tot(a1)[1]:.1f} running; V12 {tot(a12)[0]:.0f} at {tot(a12)[1]:.1f}")


def attempts(path, trdir):
    out = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("error") or float(r.get("t_end_h") or 99) > H:
            continue
        ctx = None
        tf = r.get("trace_file")
        p = os.path.join(trdir, tf) if tf else None
        if p and os.path.exists(p):
            try:
                ev = [e for e in json.load(gzip.open(p)).get("events", []) if isinstance(e, dict) and e.get("usage")]
                if ev:
                    u = ev[-1]["usage"]; ctx = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            except Exception:
                pass
        r["_ctx"] = ctx
        out.append(r)
    return out


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def stats(rs, name):
    g = [(r.get("usage") or {}).get("completion") or 0 for r in rs]
    t = [r.get("turns") or 0 for r in rs]
    c = [r["_ctx"] for r in rs if r["_ctx"]]
    w = [r.get("wall_s") or 0 for r in rs]
    stops = defaultdict(int)
    for r in rs:
        stops[r.get("stop_reason")] += 1
    return (f"| {name} | {len(rs)} | {q(g,.5)} / {q(g,.9)} | {statistics.mean(g):.0f} | {q(t,.5)} / {q(t,.9)} | "
            f"{q(c,.5)} / {q(c,.9)} ({len(c)}) | {q(w,.5):.0f} / {q(w,.9):.0f} | {dict(stops)} |")


r1 = attempts(f"{v1d}/attempts.jsonl", v1tr); r12 = attempts(v12att, v12tr)
print(f"\n(b) finished attempts by Kaggle {H} h (error records excluded)")
print("| run / set | n | generated p50 / p90 | gen mean | turns p50 / p90 | context at last turn p50 / p90 (n) | wall s p50 / p90 | stops |")
print("|---|---|---|---|---|---|---|---|")
for name, rs in (("V1 all", r1), ("V12 all", r12), ("V1 first", [r for r in r1 if r.get("attempt") == 0]),
                 ("V12 first", [r for r in r12 if r.get("attempt") == 0])):
    print(stats(rs, name))
common = {r["task_id"] for r in r1 if r.get("attempt") == 0} & {r["task_id"] for r in r12 if r.get("attempt") == 0}
cm1 = [r for r in r1 if r.get("attempt") == 0 and r["task_id"] in common]
cm12 = [r for r in r12 if r.get("attempt") == 0 and r["task_id"] in common]
print(stats(cm1, "V1 first, common tasks")); print(stats(cm12, "V12 first, common tasks"))


def correct(r):
    t = sol[r["task_id"]]; pr = r.get("pred")
    pr = json.loads(pr) if isinstance(pr, str) else pr
    if r.get("multi_test"):
        return isinstance(pr, list) and len(pr) == len(t) and all(pr[i] == t[i] for i in range(len(t)))
    return pr == t[0] if r.get("pair", 0) == 0 else None


def first(rs):
    return {r["task_id"]: correct(r) for r in rs if r.get("attempt") == 0 and r.get("pair", 0) == 0}


f1, f12 = first(r1), first(r12)
both = sorted(set(f1) & set(f12))
n1 = sum(bool(f1[t]) for t in both); n12 = sum(bool(f12[t]) for t in both)
only1 = sum(1 for t in both if f1[t] and not f12[t]); only12 = sum(1 for t in both if f12[t] and not f1[t])
print(f"\n(d) first attempts (pair 0 record, every pair right) on the {len(both)} tasks both finished by {H} h: "
      f"V1 {n1} ({100*n1/max(1,len(both)):.1f}%), V12 {n12} ({100*n12/max(1,len(both)):.1f}%); V1-only {only1}, V12-only {only12}")
print(f"tasks with a finished first attempt by {H} h: V1 {len(f1)}, V12 {len(f12)}; V1-only tasks {len(set(f1)-set(f12))}, V12-only {len(set(f12)-set(f1))}")
