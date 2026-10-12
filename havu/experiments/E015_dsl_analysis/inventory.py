"""Q1: inventory of arc-dsl primitives and stats of the 400 ARC-AGI-1 training solvers.

    .venv/bin/python -I experiments/E015_dsl_analysis/inventory.py
Writes inventory.json next to this script.
"""
import ast
import json
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, VENDOR  # noqa: E402

CATEGORIES = {
    "arithmetic_logic": "identity add subtract multiply divide invert even double halve flip equality contained both "
                        "either increment decrement crement sign positive greater toivec tojvec astuple interval",
    "set_container": "combine intersection difference dedupe order repeat size merge maximum minimum valmax valmin "
                     "argmax argmin mostcommon leastcommon initset sfilter mfilter extract totuple first last insert "
                     "remove other product pair",
    "functional_combinators": "branch compose chain matcher rbind lbind power fork apply rapply mapply papply "
                              "mpapply prapply",
    "colour": "mostcolor leastcolor colorcount colorfilter palette numcolors color recolor replace switch",
    "object_extraction_properties": "objects partition fgpartition sizefilter ofcolor asindices toindices toobject "
                                    "asobject uppermost lowermost leftmost rightmost ulcorner urcorner llcorner "
                                    "lrcorner height width shape portrait square vline hline centerofmass center "
                                    "corners index occurrences",
    "spatial_relations": "hmatching vmatching manhattan adjacent bordering position gravitate dneighbors ineighbors "
                         "neighbors",
    "patch_geometry": "shift normalize backdrop delta inbox outbox box connect shoot vfrontier hfrontier",
    "grid_transforms": "rot90 rot180 rot270 hmirror vmirror dmirror cmirror upscale hupscale vupscale downscale "
                       "hconcat vconcat crop subgrid hsplit vsplit tophalf bottomhalf lefthalf righthalf trim "
                       "compress frontiers cellwise hperiod vperiod canvas",
    "painting": "fill paint underfill underpaint cover move",
}
CAT_OF = {f: c for c, fs in CATEGORIES.items() for f in fs.split()}


def main():
    dsl_src = open(os.path.join(VENDOR, "dsl.py")).read()
    prims = [n.name for n in ast.parse(dsl_src).body if isinstance(n, ast.FunctionDef)]
    missing = [p for p in prims if p not in CAT_OF]
    assert not missing, missing
    cat_counts = Counter(CAT_OF[p] for p in prims)

    sol_src = open(os.path.join(VENDOR, "solvers.py")).read()
    sols = [n for n in ast.parse(sol_src).body if isinstance(n, ast.FunctionDef)]
    prim_set = set(prims)
    per = []
    use = Counter()       # number of solvers using each primitive
    calls_total = Counter()
    for fn in sols:
        stmts = [s for s in fn.body if isinstance(s, ast.Assign)]
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        names = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name)]
        used = {x for x in names if x in prim_set}  # primitives called or passed as function arguments
        hof = sum(1 for c in calls if isinstance(c.func, ast.Name) and CAT_OF.get(c.func.id) == "functional_combinators")
        for u in used:
            use[u] += 1
        for x in names:
            if x in prim_set:
                calls_total[x] += 1
        src_lines = ast.get_source_segment(sol_src, fn).count("\n") + 1
        per.append({"task": fn.name[6:], "lines": src_lines, "assignments": len(stmts), "calls": len(calls),
                    "distinct_primitives": len(used), "combinator_calls": hof})
    def stats(key):
        v = [p[key] for p in per]
        q = st.quantiles(v, n=20)
        return {"mean": round(st.mean(v), 2), "median": st.median(v), "p90": q[17], "p95": q[18], "max": max(v),
                "min": min(v)}
    buckets = Counter()
    for p in per:
        a = p["assignments"]
        buckets["1-3" if a <= 3 else "4-7" if a <= 7 else "8-15" if a <= 15 else "16-30" if a <= 30 else ">30"] += 1
    never_used = sorted(p for p in prims if use[p] == 0)
    cat_use = Counter()
    for p in per:
        pass
    out = {
        "n_primitives": len(prims),
        "category_counts": dict(cat_counts.most_common()),
        "categories": {c: fs.split() for c, fs in CATEGORIES.items()},
        "n_solvers": len(sols),
        "solver_stats": {k: stats(k) for k in ("lines", "assignments", "calls", "distinct_primitives",
                                               "combinator_calls")},
        "assignment_buckets": dict(buckets),
        "top_primitives_by_solver_count": use.most_common(40),
        "primitive_use_by_category": {c: sum(use[f] for f in fs.split()) for c, fs in CATEGORIES.items()},
        "n_primitives_never_used": len(never_used),
        "never_used": never_used,
        "longest_solvers": sorted(per, key=lambda p: -p["assignments"])[:5],
        "per_solver": per,
    }
    json.dump(out, open(os.path.join(HERE, "inventory.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_solver", "categories")}, indent=1))


if __name__ == "__main__":
    main()
