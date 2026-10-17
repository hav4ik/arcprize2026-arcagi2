"""E026: aggregate ds_features.jsonl into per-condition tables and the paired S80-vs-V1 comparison. No API calls.
Run: .venv/bin/python -I experiments/E026_trace_strategy/scripts/ds_stats.py"""
import json, random, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
rows = [json.loads(l) for l in open(ROOT / "experiments/E026_trace_strategy/ds_features.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["run"]].append(r)

def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else float("nan")

def mean(x):
    x = [v for v in x if v is not None]
    return st.mean(x) if x else float("nan")

def tok_of_chars(r, chars):
    cpt = r.get("chars_per_tok") or 3.0
    return chars / cpt

COND = ["NT", "S80", "S96", "V1", "V3"]
print("## Volume")
print("cond | n | acc | tokens mean/median | chars/tok")
for c in COND:
    R = by[c]
    print(f"{c} | {len(R)} | {mean([r['correct'] for r in R]):.1%} | {mean([r['tokens'] for r in R])/1e3:.1f}k / "
          f"{med([r['tokens'] for r in R])/1e3:.1f}k | {med([r['chars_per_tok'] for r in R]):.2f}")

print("\n## Hand perception in reasoning text (per attempt)")
print("cond | grid lines mean | >=10 | >=40 | coord-list lines mean | coord tuples mean | transcription share of reasoning chars "
      "(grid+coord lines) mean | est. transcription tokens mean / median")
for c in COND:
    R = by[c]
    share = [(r["grid_chars"] + r["coord_chars"]) / max(1, r["chars"]) for r in R]
    ttok = [tok_of_chars(r, r["grid_chars"] + r["coord_chars"]) for r in R]
    print(f"{c} | {mean([r['grid_lines'] for r in R]):.1f} | {mean([r['grid_lines']>=10 for r in R]):.0%} | "
          f"{mean([r['grid_lines']>=40 for r in R]):.0%} | {mean([r['coord_lines'] for r in R]):.1f} | "
          f"{mean([r['coord_tuples'] for r in R]):.0f} | {mean(share):.1%} | {mean(ttok)/1e3:.1f}k / {med(ttok)/1e3:.1f}k")

print("\n## Own transcription (grid lines + coordinate-list lines that do not quote summary/tool formats)")
print("cond | own coord lines mean | own transcription tokens mean / median | share of reasoning chars | quote lines mean | perception-code tokens mean")
for c in COND:
    R = by[c]
    ot = [tok_of_chars(r, r["grid_chars"] + r["coord_own_chars"]) for r in R]
    print(f"{c} | {mean([r['coord_own_lines'] for r in R]):.1f} | {mean(ot)/1e3:.1f}k / {med(ot)/1e3:.1f}k | "
          f"{mean([(r['grid_chars']+r['coord_own_chars'])/max(1,r['chars']) for r in R]):.1%} | {mean([r['squote_lines'] for r in R]):.1f} | "
          f"{mean([(r.get('perc_code_chars') or 0)/3 for r in R])/1e3:.2f}k")

print("\n## Summary-like statistics mentioned in reasoning: % attempts with >=1 mention; mentions per 10k chars")
keys = ["dim", "bg", "count", "obj", "hole", "diff", "sym", "cst", "audit", "noise", "uncert", "giveup", "summary_word"]
print("cond | " + " | ".join(keys))
for c in COND:
    R = by[c]
    cells = []
    for k in keys:
        a = mean([r["n_" + k] > 0 for r in R]); d = mean([r["n_" + k] / max(1, r["chars"]) * 1e4 for r in R])
        cells.append(f"{a:.0%} ({d:.1f})")
    print(c + " | " + " | ".join(cells))

print("\n## Perception in Python code (tool runs). _pre = in cells before the first solve()/transform() definition")
ck = ["c_shape", "c_colours", "c_components", "c_bbox", "c_diff", "c_d8", "c_cst", "c_print_grid", "c_test"]
print("cond | cells mean | cells before solve mean | perception cells (no solve def) mean | perc code share of code chars | " +
      " | ".join(k[2:] + " pre/any" for k in ck))
for c in COND[1:]:
    R = by[c]
    pcs = [r["perc_code_chars"] / max(1, r["code_chars"]) for r in R]
    print(f"{c} | {mean([r['n_cells'] for r in R]):.1f} | {mean([r['cells_before_solve'] for r in R]):.1f} | "
          f"{mean([r['perc_cells_all'] for r in R]):.1f} | {mean(pcs):.0%} | " +
          " | ".join(f"{mean([r[k+'_pre'] for r in R]):.0%}/{mean([r[k+'_any'] for r in R]):.0%}" for k in ck))

print("\n## First turn / time to first solve (tool runs)")
print("cond | turn-1 tokens median / mean | turn-1 share of attempt median | turn-1 grid lines mean | turn-1 >=5k tok | "
      "tokens before first solve() def median | attempts never defining solve")
for c in COND[1:]:
    R = by[c]
    print(f"{c} | {med([r['turn1_tok'] for r in R]):.0f} / {mean([r['turn1_tok'] for r in R]):.0f} | "
          f"{med([r['turn1_tok']/max(1,r['tokens']) for r in R]):.0%} | {mean([r['turn1_grid_lines'] for r in R]):.1f} | "
          f"{mean([r['turn1_tok']>=5000 for r in R]):.0%} | {med([r['tok_before_solve'] for r in R]):.0f} | "
          f"{mean([r['tok_before_solve'] is None for r in R]):.0%}")

# where do hand-transcribed lines sit, by turn index bucket (tool) and by decile (NT)
print("\n## NT: share of transcription chars by decile of the trace (token-position)")
dec = [0] * 10
for r in by["NT"]:
    for i, v in enumerate(r["perc_decile_chars"]):
        dec[i] += v
tot = sum(dec)
print(" ".join(f"{v/tot:.1%}" for v in dec))
print("NT: test-application marker position median", f"{med([r['test_apply_pos'] for r in by['NT']]):.0%}",
      "found in", f"{mean([r['test_apply_pos'] is not None for r in by['NT']]):.0%}")

# ---------- paired S80 (mean of samples) vs V1 on the same pairs ----------
def per_pair(c, f):
    d = defaultdict(list)
    for r in by[c]:
        v = f(r)
        if v is not None:
            d[(r["task_id"], r["pair"])].append(v)
    return {k: st.mean(v) for k, v in d.items()}

METRICS = {
    "tokens": lambda r: r["tokens"],
    "turn1 tokens": lambda r: r.get("turn1_tok"),
    "tokens before first solve()": lambda r: r.get("tok_before_solve"),
    "hand grid lines": lambda r: r["grid_lines"],
    "hand transcription tokens (grid+coord lines)": lambda r: tok_of_chars(r, r["grid_chars"] + r["coord_chars"]),
    "coord tuples": lambda r: r["coord_tuples"],
    "own coord-list lines (not quoting summary/tool formats)": lambda r: r["coord_own_lines"],
    "own transcription tokens (grid + own coord lines)": lambda r: tok_of_chars(r, r["grid_chars"] + r["coord_own_chars"]),
    "summary/tool-format quoting lines": lambda r: r["squote_lines"],
    "perception code tokens (cells w/o solve def, chars/3)": lambda r: (r.get("perc_code_chars") or 0) / 3.0,
    "count-word mentions": lambda r: r["n_count"],
    "perception code cells (no solve def)": lambda r: r.get("perc_cells_all"),
    "cells before first solve()": lambda r: r.get("cells_before_solve"),
    "code: colour counts (any)": lambda r: float(r.get("c_colours_any", 0)),
    "code: components (any)": lambda r: float(r.get("c_components_any", 0)),
    "code: diff (any)": lambda r: float(r.get("c_diff_any", 0)),
    "code: shape (any)": lambda r: float(r.get("c_shape_any", 0)),
    "code: d8 (any)": lambda r: float(r.get("c_d8_any", 0)),
    "code: print grid (any)": lambda r: float(r.get("c_print_grid_any", 0)),
    "code before solve: diff": lambda r: float(r.get("c_diff_pre", 0)),
    "code before solve: components": lambda r: float(r.get("c_components_pre", 0)),
    "code before solve: colour counts": lambda r: float(r.get("c_colours_pre", 0)),
    "test-vs-train audit (regex)": lambda r: float(r["n_audit"] > 0),
}

def boot(diffs, n=5000, seed=0):
    rng = random.Random(seed)
    tasks = defaultdict(list)
    for (t, p), v in diffs.items():
        tasks[t].append(v)
    items = list(tasks.values())
    ms = []
    for _ in range(n):
        s = rng.choices(items, k=len(items))
        flat = [x for g in s for x in g]
        ms.append(st.mean(flat))
    ms.sort()
    return ms[int(0.025 * n)], ms[int(0.975 * n)]

for A, B in [("S80", "V1"), ("S80", "V3"), ("S96", "V1")]:
    print(f"\n## Paired {A} (mean of samples per pair) vs {B}, same pairs")
    print(f"metric | {A} mean | {B} mean | diff | 95% CI (task bootstrap) | {A} median | {B} median")
    for name, f in METRICS.items():
        a, b = per_pair(A, f), per_pair(B, f)
        ks = sorted(set(a) & set(b))
        if not ks:
            continue
        d = {k: b[k] - a[k] for k in ks}
        lo, hi = boot(d)
        print(f"{name} | {st.mean(a[k] for k in ks):.2f} | {st.mean(b[k] for k in ks):.2f} | {st.mean(d.values()):+.2f} | "
              f"[{lo:+.2f}, {hi:+.2f}] | {st.median(a[k] for k in ks):.1f} | {st.median(b[k] for k in ks):.1f}  (n={len(ks)})")

# NT vs tool on the same pairs: transcription
print("\n## NT vs S80 vs V1, same pairs: hand transcription tokens (mean per pair)")
f = METRICS["hand transcription tokens (grid+coord lines)"]
a, b, c = per_pair("NT", f), per_pair("S80", f), per_pair("V1", f)
ks = sorted(set(a) & set(b) & set(c))
print(f"n={len(ks)} NT {st.mean(a[k] for k in ks)/1e3:.1f}k  S80 {st.mean(b[k] for k in ks)/1e3:.1f}k  V1 {st.mean(c[k] for k in ks)/1e3:.1f}k")

# by outcome
print("\n## Hand transcription tokens by outcome (mean)")
for cnd in COND:
    for ok in (True, False):
        R = [r for r in by[cnd] if r["correct"] == ok]
        print(cnd, "correct" if ok else "wrong", len(R), f"{mean([tok_of_chars(r, r['grid_chars']+r['coord_chars']) for r in R])/1e3:.1f}k",
              f"share {mean([(r['grid_chars']+r['coord_chars'])/max(1,r['chars']) for r in R]):.1%}")
