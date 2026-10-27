"""E058: image arm vs a text-only baseline on the same tasks; task-weighted pass@2/pass@1, paired bootstrap over tasks.

Usage: compare.py IMAGE.jsonl[,MORE.jsonl] BASE.jsonl[,MORE.jsonl] [--json]
Several files per side are merged (latest successful record per task/pair/attempt; API errors ignored). A task counts
only when both attempts of all its pairs exist on both sides.
"""
import json
import random
import sys
from collections import defaultdict


def load(paths):
    out = {}
    for path in paths.split(","):
        for line in open(path):
            r = json.loads(line)
            if not r.get("error"):
                out[r["task_id"], int(r["pair"]), int(r["attempt"])] = r
    return out


def ok(m, t, p, a):
    return bool(m.get((t, p, a), {}).get("correct"))


def complete(m, t, pairs):
    return all((t, p, a) in m for p in pairs for a in (0, 1))


def task_scores(m, t, pairs):
    s2 = sum(ok(m, t, p, 0) or ok(m, t, p, 1) for p in pairs) / len(pairs)
    s1 = sum((ok(m, t, p, 0) + ok(m, t, p, 1)) / 2 for p in pairs) / len(pairs)
    toks = defaultdict(int)
    for (tt, p, a), r in m.items():
        if tt == t:
            toks[a] += r["usage"]["completion"]
    cost = sum(r.get("est_cost_usd", 0) or 0 for (tt, p, a), r in m.items() if tt == t)
    return s2, s1, sum(toks.values()) / max(1, len(toks)), cost


def compare(img_paths, base_paths):
    img, base = load(img_paths), load(base_paths)
    pairs = defaultdict(set)
    for t, p, a in base:
        pairs[t].add(p)
    tasks = sorted(t for t in {t for t, p, a in img} if complete(img, t, pairs[t]) and complete(base, t, pairs[t]))
    res = {"tasks": len(tasks)}
    if not tasks:
        return res
    for lab, m in (("text", base), ("image", img)):
        v = [task_scores(m, t, sorted(pairs[t])) for t in tasks]
        res[lab] = dict(pass2=sum(x[0] for x in v) / len(v), pass1=sum(x[1] for x in v) / len(v),
                        tok=sum(x[2] for x in v) / len(v), cost=sum(x[3] for x in v), per_task=[x[0] for x in v])
    d = [a - b for a, b in zip(res["image"]["per_task"], res["text"]["per_task"])]
    random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
    res.update(delta=sum(d) / len(d), lo=bs[100], hi=bs[3899], better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d))
    return res


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2])
    if "--json" in sys.argv:
        print(json.dumps({k: v for k, v in r.items() if k not in ("text", "image")} |
                         {k: {kk: vv for kk, vv in r[k].items() if kk != "per_task"} for k in ("text", "image") if k in r}))
        sys.exit()
    print(f"{r['tasks']} tasks complete on both sides")
    if r["tasks"]:
        print(f"{'':12s} {'pass@2':>7s} {'pass@1':>7s} {'out tok/task-attempt':>21s}  cost")
        for lab in ("text", "image"):
            x = r[lab]
            print(f"{lab:12s} {x['pass2']:7.1%} {x['pass1']:7.1%} {x['tok'] / 1000:17.1f}k  ${x['cost']:.2f}")
        print(f"pass@2 image - text: {r['delta']:+.1%} [95% CI {r['lo']:+.1%}, {r['hi']:+.1%}]; "
              f"tasks better {r['better']}, worse {r['worse']}")
