"""Offline test of a cheap, LLM-free 'invariant critic': learn simple invariants that hold on ALL train pairs and flag
a submitted test answer that violates one. Precision/recall against correctness on the 835 attempts. No API."""
import json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/E017_reasoning_efficiency"
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"
RUNS = {"sub80s40": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
        "sub96n": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n.jsonl"}

def bg(a):
    v, c = np.unique(a, return_counts=True); return v[c.argmax()]

def sym(a):
    s = set()
    if np.array_equal(a, a[:, ::-1]): s.add("lr")
    if np.array_equal(a, a[::-1]): s.add("ud")
    if a.shape[0] == a.shape[1] and np.array_equal(a, a.T): s.add("tr")
    return s

def checks(pairs):
    """Each check: name -> fn(inp, out) -> bool. Kept if True on every train pair."""
    C = {
        "same_shape": lambda i, o: i.shape == o.shape,
        "palette_subset_input": lambda i, o: set(np.unique(o)) <= set(np.unique(i)),
        "nonbg_input_cells_kept": lambda i, o: i.shape == o.shape and bool((o[i != bg(i)] == i[i != bg(i)]).all()),
        "only_bg_changes_or_none": lambda i, o: i.shape == o.shape and bool((i[i != o] == bg(i)).all()),
        "some_change": lambda i, o: i.shape != o.shape or bool((i != o).any()),
        "colour_counts_conserved": lambda i, o: sorted(np.unique(i, return_counts=True)[1].tolist()) == sorted(np.unique(o, return_counts=True)[1].tolist()) and set(np.unique(i)) == set(np.unique(o)),
        "bg_preserved_as_bg_colour": lambda i, o: bg(i) == bg(o),
        "num_colours_le_input": lambda i, o: len(np.unique(o)) <= len(np.unique(i)),
        "out_smaller_or_equal": lambda i, o: o.size <= i.size,
        "out_larger_or_equal": lambda i, o: o.size >= i.size,
    }
    for s in ("lr", "ud", "tr"):
        C[f"out_sym_{s}"] = (lambda s: lambda i, o: s in sym(o))(s)
    shapes = {np.array(p["output"]).shape for p in pairs}
    keep = {k: f for k, f in C.items() if all(f(np.array(p["input"]), np.array(p["output"])) for p in pairs)}
    const_shape = shapes.pop() if len(shapes) == 1 else None
    return keep, const_shape

def flags(task, pair, pred):
    keep, const_shape = checks(task["train"])
    i, o = np.array(task["test"][pair]["input"]), np.array(pred)
    fl = [k for k, f in keep.items() if not _safe(f, i, o)]
    if const_shape and len(task["train"]) >= 2 and o.shape != const_shape and "same_shape" not in keep:
        fl.append("const_out_shape")
    return fl

def _safe(f, i, o):
    try:
        return f(i, o)
    except Exception:
        return False

def main():
    tasks = {}
    recs = {}
    for run, fn in RUNS.items():
        for l in open(ROOT / "evals/results" / fn):
            r = json.loads(l)
            if "error" in r:
                continue
            recs[(run, r["task_id"], r["pair"], r["attempt"])] = r
    stat = collections.Counter(); per_check = collections.defaultdict(collections.Counter); flagged_rows = []
    for k, r in recs.items():
        if r.get("pred") is None:
            continue
        t = tasks.setdefault(r["task_id"], json.loads((TASKS / f"{r['task_id']}.json").read_text()))
        pred = r["pred"] if isinstance(r["pred"], list) else json.loads(r["pred"])
        fl = flags(t, r["pair"], pred)
        ok = bool(r["correct"])
        stat[(ok, bool(fl))] += 1
        for f in fl:
            per_check[f][ok] += 1
        if fl:
            flagged_rows.append({"key": list(k), "correct": ok, "flags": fl})
    tp, fp = stat[(False, True)], stat[(True, True)]
    print("answers:", sum(stat.values()), " wrong:", stat[(False, True)] + stat[(False, False)], " correct:", stat[(True, True)] + stat[(True, False)])
    print(f"flagged: {tp + fp}; of which wrong {tp} (precision {tp / max(1, tp + fp):.2f}); recall on wrong {tp / max(1, tp + stat[(False, False)]):.2f}; correct answers flagged {fp}")
    for c, v in sorted(per_check.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {c}: flags wrong {v[False]}, flags correct {v[True]}")
    json.dump(flagged_rows, open(OUT / "invariant_flags.json", "w"))

main()
