"""Q4 add-on: does any of Hodel's 400 ARC-AGI-1 training solvers, applied as-is, solve an ARC-AGI-2 eval task?

    .venv/bin/python -I experiments/E015_dsl_analysis/solver_transfer.py
Writes solver_transfer.json. A solver "fits" a task if it reproduces every train output; "solves" if it also
reproduces every test output.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, as_tuple, call_with_timeout, import_dsl, load_arc2_eval  # noqa: E402

_s = None
EV = None


def run(name):
    global _s, EV
    if _s is None:
        _, _s, _ = import_dsl()
        EV = load_arc2_eval()
    fn = getattr(_s, name)
    hits = []
    for tid, t in EV.items():
        try:
            if all(call_with_timeout(fn, as_tuple(p["input"]), 2) == as_tuple(p["output"]) for p in t["train"]):
                test = all(call_with_timeout(fn, as_tuple(p["input"]), 2) == as_tuple(p["output"]) for p in t["test"])
                hits.append((tid, test))
        except Exception:  # noqa: BLE001
            pass
    return name, hits


def main():
    _, s, _ = import_dsl()
    names = sorted(n for n in dir(s) if n.startswith("solve_"))
    with Pool(20) as pool:
        res = pool.map(run, names, chunksize=4)
    hits = [(n, t, ok) for n, h in res for t, ok in h]
    out = {"solvers": len(names), "fits": hits, "tasks_fit": len({t for _, t, _ in hits}),
           "tasks_solved": len({t for _, t, ok in hits if ok})}
    json.dump(out, open(os.path.join(HERE, "solver_transfer.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
