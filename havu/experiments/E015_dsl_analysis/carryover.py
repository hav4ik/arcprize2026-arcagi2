"""Q2: ARC-AGI-1 -> ARC-AGI-2 carry-over of Hodel's solvers, and ARC-1 overlap with the ARC-AGI-2 eval set.

    .venv/bin/python -I experiments/E015_dsl_analysis/carryover.py [--workers 16]
Writes carryover.json next to this script.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HERE, ROOT, as_tuple, call_with_timeout, import_dsl, load_arc1, load_arc2_eval,  # noqa: E402
                    load_arc2_train)

sys.path.insert(0, os.path.join(ROOT, "data_tools"))
from contamination_check import canon  # noqa: E402

_solvers = None


def run_solver(args):
    """Return (n_pairs, n_correct, error) for solver `name` on the given pairs."""
    name, pairs = args
    global _solvers
    if _solvers is None:
        _, _solvers, _ = import_dsl()
    fn = getattr(_solvers, name)
    ok, err = 0, None
    for p in pairs:
        try:
            if call_with_timeout(fn, as_tuple(p["input"]), 5) == as_tuple(p["output"]):
                ok += 1
        except Exception as e:  # noqa: BLE001 - solver errors are data
            err = type(e).__name__
    return len(pairs), ok, err


def pair_key(p):
    return json.dumps([p["input"], p["output"]])


def compare_task(t1, t2):
    """Describe how the ARC-AGI-2 version differs from the ARC-AGI-1 version."""
    a = [pair_key(p) for p in t1["train"] + t1["test"]]
    b = [pair_key(p) for p in t2["train"] + t2["test"]]
    if a == b:
        return "identical"
    if sorted(a) == sorted(b):
        return "same_pairs_reshuffled"
    sa, sb = set(a), set(b)
    if sa <= sb:
        return "pairs_added"
    if sb <= sa:
        return "pairs_removed"
    if sa & sb:
        return "some_pairs_changed"
    return "all_pairs_changed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    _, solvers, _ = import_dsl()
    solver_ids = sorted(n[6:] for n in dir(solvers) if n.startswith("solve_"))
    a1t, a1e = load_arc1("training"), load_arc1("evaluation")
    a2t, a2e = load_arc2_train(), load_arc2_eval()
    out = {"n_solvers": len(solver_ids), "n_arc1_train": len(a1t), "n_arc1_eval": len(a1e),
           "n_arc2_train": len(a2t), "n_arc2_eval": len(a2e)}
    out["arc2_train_ids_in_arc1_train"] = len(set(a2t) & set(a1t))
    out["arc2_train_ids_in_arc1_eval"] = len(set(a2t) & set(a1e))
    out["arc2_eval_ids_in_arc1_train"] = sorted(set(a2e) & set(a1t))
    out["arc2_eval_ids_in_arc1_eval"] = sorted(set(a2e) & set(a1e))
    out["arc2_eval_ids_in_arc2_train"] = sorted(set(a2e) & set(a2t))
    shared = sorted(set(a2t) & set(solver_ids))
    out["arc2_train_with_hodel_solver"] = len(shared)

    with Pool(args.workers) as pool:
        # Solver on its own ARC-AGI-1 task (baseline) and on the ARC-AGI-2 version of the same ID.
        base = pool.map(run_solver, [(f"solve_{t}", a1t[t]["train"] + a1t[t]["test"]) for t in solver_ids])
        v2 = pool.map(run_solver, [(f"solve_{t}", a2t[t]["train"] + a2t[t]["test"]) for t in shared])
    base_ok = {t for t, (n, k, _) in zip(solver_ids, base) if n == k}
    out["arc1_baseline_solvers_passing_all_pairs"] = len(base_ok)
    rows = []
    for t, (n, k, err) in zip(shared, v2):
        rows.append({"task": t, "diff": compare_task(a1t[t], a2t[t]), "pairs": n, "correct": k, "pass": n == k,
                     "error": err, "arc1_pass": t in base_ok})
    out["carryover_pass_all_pairs"] = sum(r["pass"] for r in rows)
    out["carryover_pass_rate"] = round(out["carryover_pass_all_pairs"] / len(rows), 4)
    out["carryover_pair_accuracy"] = round(sum(r["correct"] for r in rows) / sum(r["pairs"] for r in rows), 4)
    out["diff_kinds"] = dict(Counter(r["diff"] for r in rows))
    out["pass_by_diff_kind"] = {d: f"{sum(r['pass'] for r in rows if r['diff'] == d)}/"
                                   f"{sum(1 for r in rows if r['diff'] == d)}" for d in out["diff_kinds"]}
    out["carryover_failures"] = [r for r in rows if not r["pass"]]

    # Content overlap of the 120 ARC-AGI-2 eval tasks with ARC-AGI-1 (train+eval), D8 + colour-permutation canon.
    idx = defaultdict(set)
    for split, ds in (("t", a1t), ("e", a1e)):
        for tid, t in ds.items():
            for p in t["train"] + t["test"]:
                for g in (p["input"], p["output"]):
                    c = canon(g)
                    if c is not None:
                        idx[c].add(f"{split}:{tid}")
    ov = []
    for tid, t in a2e.items():
        grids = [g for p in t["train"] + t["test"] for g in (p["input"], p["output"])]
        keys = [canon(g) for g in grids]
        keys = [k for k in keys if k is not None]
        hits = Counter()
        n_hit = 0
        for k in keys:
            if k in idx:
                n_hit += 1
                for s in idx[k]:
                    hits[s] += 1
        if n_hit:
            best, bc = hits.most_common(1)[0]
            ov.append({"task": tid, "grids": len(keys), "grids_matching_arc1": n_hit, "best_arc1_task": best,
                       "best_task_shared_grids": bc})
    out["arc2_eval_tasks_with_any_arc1_grid_match"] = len(ov)
    out["arc2_eval_tasks_ge50pct_grids_from_one_arc1_task"] = sum(
        1 for o in ov if o["best_task_shared_grids"] >= 0.5 * o["grids"])
    out["arc2_eval_overlap_rows"] = ov

    # Same check for ARC-AGI-2 training tasks NOT sharing an ID with ARC-AGI-1 (how much of the "new" training
    # material is a reskin of ARC-1).
    new_tr = [t for t in a2t if t not in a1t and t not in a1e]
    n_dup = 0
    for tid in new_tr:
        t = a2t[tid]
        keys = [canon(g) for p in t["train"] + t["test"] for g in (p["input"], p["output"])]
        keys = [k for k in keys if k is not None]
        hits = Counter(s for k in keys if k in idx for s in idx[k])
        if hits and hits.most_common(1)[0][1] >= 0.5 * len(keys):
            n_dup += 1
    out["arc2_train_new_ids"] = len(new_tr)
    out["arc2_train_new_ids_ge50pct_grids_from_one_arc1_task"] = n_dup

    json.dump(out, open(os.path.join(HERE, "carryover.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("carryover_failures", "arc2_eval_overlap_rows")},
                     indent=1))
    print("failures:", [(r["task"], r["diff"], f"{r['correct']}/{r['pairs']}", r["error"])
                        for r in out["carryover_failures"]])
    print("eval overlap:", ov)


if __name__ == "__main__":
    main()
