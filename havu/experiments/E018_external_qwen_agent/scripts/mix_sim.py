"""E018: simulated pass@2 of mixed attempt pairs (tools medium 48k + no-tool xhigh capped), from E012/E008 files."""
import json, statistics, sys
from pathlib import Path
R = Path(sys.argv[1])
def load(names):
    out = {}
    for n in names:
        for l in (R / n).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return out
tools = load(["qwen_tools_medium_48k_dev_s24.jsonl", "qwen_tools_medium_48k_full_v3.jsonl"])
nt = load(["qwen_nvfp4_gen_xhigh_240k_t1.jsonl"])
pairs = sorted({(t, p) for t, p, a in tools} & {(t, p) for t, p, a in nt})
def tok_tools(r): return r["usage"]["completion"]
def T(t, p, a): r = tools.get((t, p, a)); return (bool(r and r["correct"]), tok_tools(r) if r else 0)
def N(cap):
    def f(t, p, a):
        r = nt.get((t, p, a))
        if not r: return (False, 0)
        k = r["completion_tokens"]; return (bool(r["correct"]) and k <= cap, min(k, cap))
    return f
def score(fa, fb):
    tasks = {}; toks = []; pw = []
    for t, p in pairs:
        vals = []
        for a in (0, 1):
            ca, ka = fa(t, p, a); cb, kb = fb(t, p, 1 - a) if fb else (False, 0)
            vals.append(ca or cb); toks.append(ka + kb)
        v = statistics.mean(vals); tasks.setdefault(t, []).append(v); pw.append(v)
    return 100 * statistics.mean(statistics.mean(v) for v in tasks.values()), 100 * statistics.mean(pw), statistics.mean(toks) / 1000
print(f"pairs {len(pairs)}")
rows = [("tools x1 (pass@1)", T, None), ("tools x2", T, T)]
for cap in (48000, 64000, 96000):
    rows += [(f"no-tool xhigh {cap//1000}k x1", N(cap), None), (f"no-tool xhigh {cap//1000}k x2", N(cap), N(cap)),
             (f"tools + no-tool {cap//1000}k", T, N(cap))]
for name, a, b in rows:
    tw, pw, k = score(a, b)
    print(f"{name:28s} task-w {tw:5.1f}  pair-w {pw:5.1f}  tokens/pair {k:5.1f}k")
import random
def per_task(fa, fb):
    d = {}
    for t, p in pairs:
        v = statistics.mean([(fa(t, p, a)[0] or (fb(t, p, 1 - a)[0] if fb else False)) for a in (0, 1)])
        d.setdefault(t, []).append(v)
    return {t: statistics.mean(v) for t, v in d.items()}
base = per_task(T, T); ids = sorted(base); rng = random.Random(0)
for cap in (48000, 64000):
    m = per_task(T, N(cap)); diff = {t: m[t] - base[t] for t in ids}
    bs = sorted(statistics.mean(diff[rng.choice(ids)] for _ in ids) for _ in range(4000))
    print(f"tools+no-tool {cap//1000}k minus tools x2: {100*statistics.mean(diff.values()):+.1f} [{100*bs[100]:+.1f}, {100*bs[3899]:+.1f}] (task-weighted, 95% bootstrap over tasks)")
