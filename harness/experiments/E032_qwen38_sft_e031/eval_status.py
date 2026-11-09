"""E032 eval progress per arm and pair set.

pass@1 = attempt #0 (the first pass); pass@2 = either of attempts #0/#1 correct, over pairs that have both; paired
counts compare the arms on the pairs both have finished.

  python3 eval_status.py
"""
import json
import statistics as st
from pathlib import Path

R = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")


def load(model, s):
    out = {}
    for part in ([s] if s != "k93" else ["k93", "k93a", "k93b"]):  # k93 was split across the two boxes
        f = R / f"tools_persistent_grid_lean2sub_{model}_low_e032_{part}.jsonl"
        if f.exists():
            out.update({(r["task_id"], r["pair"], r.get("attempt", 0)): r for r in map(json.loads, open(f))})
    return out


def ok(r):
    return r is not None and r.get("correct") is True


for s in ("dev56", "dev56_effmed", "dev56_effmin", "k93", "rest"):  # _effmed / _effmin: Qwen tier medium / minimal
    arms = {m: load(m, s) for m in ("e032", "base")}
    if not any(arms.values()):
        continue
    a0, both = {}, {}
    for m, d in arms.items():
        if not d:
            continue
        rs = [r for k, r in d.items() if k[2] == 0]
        a0[m] = {k[:2]: r for k, r in d.items() if k[2] == 0}
        both[m] = {k[:2]: (r, d[(*k[:2], 1)]) for k, r in d.items() if k[2] == 0 and (*k[:2], 1) in d}
        n = len(rs)
        toks = [r["usage"]["completion"] for r in d.values() if r.get("usage")]
        line = (f"{s:12s} {m:5s} pass@1 {sum(map(ok, rs)):3d}/{n:3d} ({100 * sum(map(ok, rs)) / max(n, 1):5.1f}%)"
                f"  cap {sum(r.get('stop_reason') == 'output_budget' for r in d.values()):3d}"
                f"  errors {sum(bool(r.get('error')) for r in d.values())}"
                f"  out tok mean {st.mean(toks) / 1e3 if toks else 0:5.1f}k median {st.median(toks) / 1e3 if toks else 0:5.1f}k")
        if both[m]:
            p2 = sum(ok(x) or ok(y) for x, y in both[m].values())
            line += f"  | pass@2 {p2}/{len(both[m])} ({100 * p2 / len(both[m]):.1f}%)"
        print(line)
    if len(a0) == 2:
        common = set(a0["e032"]) & set(a0["base"])
        if common:
            e = sum(ok(a0["e032"][k]) for k in common); b = sum(ok(a0["base"][k]) for k in common)
            eo = sum(ok(a0["e032"][k]) and not ok(a0["base"][k]) for k in common)
            bo = sum(ok(a0["base"][k]) and not ok(a0["e032"][k]) for k in common)
            print(f"{s:12s} paired pass@1 on {len(common)}: e032 {e} vs base {b} (e032-only {eo}, base-only {bo})")
        common2 = set(both.get("e032", {})) & set(both.get("base", {}))
        if common2:
            e = sum(ok(x) or ok(y) for k in common2 for x, y in [both["e032"][k]])
            b = sum(ok(x) or ok(y) for k in common2 for x, y in [both["base"][k]])
            print(f"{s:12s} paired pass@2 on {len(common2)}: e032 {e} vs base {b}")
