"""Q4: brute-force breadth-first search over compositions (depth <= 3) of arc-dsl grid->grid transforms.

Each transform is an arc-dsl expression with its non-grid arguments bound (colours come from the task's train
palette). The search is bottom-up with observational-equivalence pruning: a state is the tuple of transformed
train inputs, duplicates are dropped, and a program "fits" when the state equals the train outputs. The first
fitting program found is then checked on the test pairs. Every task gets a fixed wall-clock budget.

    .venv/bin/python -I experiments/E015_dsl_analysis/search.py --set arc2_eval --budget 10 --workers 20
    .venv/bin/python -I experiments/E015_dsl_analysis/search.py --set arc1_train --budget 3 --workers 20
Writes search_<set>.json next to this script.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, as_tuple, import_dsl, load_arc1, load_arc2_eval  # noqa: E402

D = None


def build_transforms(palette):
    """List of (name, fn) grid->grid transforms; colour-parameterised ones use the task palette."""
    d = D
    T, F = True, False
    tr = []
    geo = {"rot90": d.rot90, "rot180": d.rot180, "rot270": d.rot270, "hmirror": d.hmirror, "vmirror": d.vmirror,
           "dmirror": d.dmirror, "cmirror": d.cmirror}
    for n, f in geo.items():
        tr.append((n, f))
    for n in ("trim", "compress", "tophalf", "bottomhalf", "lefthalf", "righthalf"):
        tr.append((n, getattr(d, n)))
    # symmetry completion and tiling
    for n, f in list(geo.items()) + [("id", d.identity)]:
        tr.append((f"hconcat(g,{n}(g))", lambda g, f=f: d.hconcat(g, f(g))))
        tr.append((f"vconcat(g,{n}(g))", lambda g, f=f: d.vconcat(g, f(g))))
        if n != "id":
            tr.append((f"hconcat({n}(g),g)", lambda g, f=f: d.hconcat(f(g), g)))
            tr.append((f"vconcat({n}(g),g)", lambda g, f=f: d.vconcat(f(g), g)))
            tr.append((f"underpaint(g,asobject({n}(g)))", lambda g, f=f: d.underpaint(g, d.asobject(f(g)))))
    for k in (2, 3, 4):
        tr.append((f"upscale(g,{k})", lambda g, k=k: d.upscale(g, k)))
    for k in (2, 3):
        tr.append((f"downscale(g,{k})", lambda g, k=k: d.downscale(g, k)))
        tr.append((f"hupscale(g,{k})", lambda g, k=k: d.hupscale(g, k)))
        tr.append((f"vupscale(g,{k})", lambda g, k=k: d.vupscale(g, k)))
    # overlay of two halves / split parts
    splits = {"lr": lambda g: (d.lefthalf(g), d.righthalf(g)), "tb": lambda g: (d.tophalf(g), d.bottomhalf(g)),
              "hs2": lambda g: tuple(d.hsplit(g, 2)), "vs2": lambda g: tuple(d.vsplit(g, 2))}
    for sn, sp in splits.items():
        for c in palette:
            tr.append((f"cellwise({sn},{c})", lambda g, sp=sp, c=c: d.cellwise(*sp(g), c)))
    # colour ops
    for a in palette:
        for b in palette:
            if a != b:
                tr.append((f"replace(g,{a},{b})", lambda g, a=a, b=b: d.replace(g, a, b)))
                if a < b:
                    tr.append((f"switch(g,{a},{b})", lambda g, a=a, b=b: d.switch(g, a, b)))
    # object-based ops
    for uv in (T, F):
        for dg in (F, T):
            tag = f"{'T' if uv else 'F'}{'T' if dg else 'F'}T"
            objs = lambda g, uv=uv, dg=dg: d.objects(g, uv, dg, T)
            tr.append((f"subgrid(argmax(objects{tag},size))", lambda g, o=objs: d.subgrid(d.argmax(o(g), d.size), g)))
            tr.append((f"subgrid(argmin(objects{tag},size))", lambda g, o=objs: d.subgrid(d.argmin(o(g), d.size), g)))
            tr.append((f"subgrid(argmax(objects{tag},numcolors))",
                       lambda g, o=objs: d.subgrid(d.argmax(o(g), d.numcolors), g)))
            tr.append((f"cover(g,argmin(objects{tag},size))", lambda g, o=objs: d.cover(g, d.argmin(o(g), d.size))))
            tr.append((f"cover(g,argmax(objects{tag},size))", lambda g, o=objs: d.cover(g, d.argmax(o(g), d.size))))
            for c in palette:
                tr.append((f"fill(g,{c},mapply(delta,objects{tag}))",
                           lambda g, o=objs, c=c: d.fill(g, c, d.mapply(d.delta, o(g)))))
                tr.append((f"underfill(g,{c},mapply(outbox,objects{tag}))",
                           lambda g, o=objs, c=c: d.underfill(g, c, d.mapply(d.outbox, o(g)))))
                tr.append((f"fill(g,{c},argmax(objects{tag},size))",
                           lambda g, o=objs, c=c: d.fill(g, c, d.argmax(o(g), d.size))))
                tr.append((f"fill(g,{c},argmin(objects{tag},size))",
                           lambda g, o=objs, c=c: d.fill(g, c, d.argmin(o(g), d.size))))
    tr.append(("subgrid(merge(fgpartition(g)))", lambda g: d.subgrid(d.merge(d.fgpartition(g)), g)))
    for c in palette:
        # enclosed background regions (Hodel 00d62c1b idiom)
        tr.append((f"fill(g,{c},enclosed_bg)", lambda g, c=c: d.fill(g, c, d.mfilter(
            d.colorfilter(d.objects(g, T, F, F), d.mostcolor(g)), lambda o: not d.bordering(o, g)))))
        for fn in ("hfrontier", "vfrontier"):
            tr.append((f"underfill(g,{c},mapply({fn},ofcolor(g,{c})))",
                       lambda g, c=c, fn=getattr(d, fn): d.underfill(g, c, d.mapply(fn, d.ofcolor(g, c)))))
        tr.append((f"underfill(g,{c},connect_aligned(ofcolor(g,{c})))", lambda g, c=c: d.underfill(g, c, d.mfilter(
            d.prapply(d.connect, d.ofcolor(g, c), d.ofcolor(g, c)), lambda p: d.vline(p) or d.hline(p)))))
    return tr


def valid(g):
    return isinstance(g, tuple) and 0 < len(g) <= 30 and isinstance(g[0], tuple) and 0 < len(g[0]) <= 30


def apply_all(fn, state):
    out = []
    for g in state:
        r = fn(g)
        if not valid(r):
            return None
        out.append(r)
    return tuple(out)


def solve_task(args):
    tid, task, budget, max_depth = args
    global D
    if D is None:
        D, _, _ = import_dsl()
    t0 = time.time()
    xs = tuple(as_tuple(p["input"]) for p in task["train"])
    ys = tuple(as_tuple(p["output"]) for p in task["train"])
    palette = sorted({v for p in task["train"] for g in (p["input"], p["output"]) for r in g for v in r})
    trs = build_transforms(palette)
    seen = {xs}
    frontier = [(xs, ())]
    evals = 0
    depth_done = 0
    found = None
    timed_out = False
    for depth in range(1, max_depth + 1):
        nxt = []
        for state, prog in frontier:
            for i, (_, fn) in enumerate(trs):
                if time.time() - t0 > budget:
                    timed_out = True
                    break
                evals += 1
                try:
                    s2 = apply_all(fn, state)
                except Exception:  # noqa: BLE001
                    continue
                if s2 is None or s2 in seen:
                    continue
                seen.add(s2)
                if s2 == ys:
                    found = prog + (i,)
                    break
                nxt.append((s2, prog + (i,)))
            if found or timed_out:
                break
        if found or timed_out:
            break
        depth_done = depth
        frontier = nxt
    res = {"task": tid, "n_transforms": len(trs), "evals": evals, "depth_completed": depth_done,
           "timed_out": timed_out, "seconds": round(time.time() - t0, 2), "fit_train": found is not None}
    if found:
        res["program"] = [trs[i][0] for i in found]
        ok = True
        for p in task["test"]:
            g = as_tuple(p["input"])
            try:
                for i in found:
                    g = trs[i][1](g)
                ok = ok and g == as_tuple(p["output"])
            except Exception:  # noqa: BLE001
                ok = False
        res["test_correct"] = ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="arc2_eval", choices=["arc2_eval", "arc1_train", "arc1_eval"])
    ap.add_argument("--budget", type=float, default=10.0, help="seconds per task")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    data = load_arc2_eval() if args.set == "arc2_eval" else load_arc1(args.set.split("_")[1].replace(
        "train", "training").replace("eval", "evaluation"))
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(solve_task, [(t, data[t], args.budget, args.depth) for t in sorted(data)], chunksize=1)
    n = len(rows)
    fit = [r for r in rows if r["fit_train"]]
    ok = [r for r in fit if r.get("test_correct")]
    summ = {"set": args.set, "tasks": n, "budget_s": args.budget, "max_depth": args.depth,
            "cpu_seconds": round(sum(r["seconds"] for r in rows), 1), "wall_seconds": round(time.time() - t0, 1),
            "fit_train": len(fit), "test_correct": len(ok), "test_correct_pct": round(100 * len(ok) / n, 2),
            "depth_completed_hist": {k: sum(1 for r in rows if r["depth_completed"] == k) for k in range(0, 4)},
            "median_transforms": sorted(r["n_transforms"] for r in rows)[n // 2],
            "solved": [(r["task"], r["program"], r["test_correct"]) for r in fit]}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(HERE, f"search_{args.set}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
