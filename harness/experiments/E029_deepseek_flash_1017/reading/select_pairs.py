"""Select OLD/NEW sub80s40 trace pairs for close reading (E029). No API calls.
Usage: .venv/bin/python -I select_pairs.py  -> writes selection.json next to this script."""
import json
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
OLD = RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl"
NEW = RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e029_flash1017_sub80s40.jsonl"

def load(f):
    d = defaultdict(dict)
    for l in open(f):
        r = json.loads(l)
        if "error" in r: continue
        d[(r["task_id"], int(r["pair"]))][int(r["attempt"])] = r
    return d

old, new = load(OLD), load(NEW)
tok = lambda r: r["usage"]["completion"]
rows = []
for k in sorted(set(old) & set(new)):
    o, n = old[k], new[k]
    rows.append(dict(k=k, o=o, n=n,
        oc=[a for a in o if o[a]["correct"]], nc=[a for a in n if n[a]["correct"]]))

def pick_attempt(r, want_o, want_n):
    """prefer the same attempt index satisfying both predicates; else best available."""
    for a in (0, 1):
        if a in r["o"] and a in r["n"] and want_o(r["o"][a]) and want_n(r["n"][a]):
            return a, a
    ao = next((a for a in sorted(r["o"]) if want_o(r["o"][a])), None)
    an = next((a for a in sorted(r["n"]) if want_n(r["n"][a])), None)
    return ao, an

sel, used, used_tasks = [], set(), set()  # one pair per task, for diversity
def add(group, r, ao, an, why):
    used.add(r["k"]); used_tasks.add(r["k"][0])
    sel.append(dict(group=group, task=r["k"][0], pair=r["k"][1], old_attempt=ao, new_attempt=an,
        old_tokens=tok(r["o"][ao]), new_tokens=tok(r["n"][an]),
        old_correct=r["o"][ao]["correct"], new_correct=r["n"][an]["correct"],
        old_pair_acc=f"{len(r['oc'])}/{len(r['o'])}", new_pair_acc=f"{len(r['nc'])}/{len(r['n'])}",
        old_trace=r["o"][ao]["trace_path"], new_trace=r["n"][an]["trace_path"], why=why))

C = lambda x: x["correct"]
# G1 SAVE: all attempts correct in both models; rank by mean(old tok) - mean(new tok); same attempt idx (0 preferred)
g1 = [r for r in rows if len(r["oc"]) == len(r["o"]) and len(r["nc"]) == len(r["n"])]
for r in g1:
    r["save"] = sum(map(tok, r["o"].values())) / len(r["o"]) - sum(map(tok, r["n"].values())) / len(r["n"])
for r in sorted(g1, key=lambda r: -r["save"])[:12]:
    ao, an = pick_attempt(r, C, C)
    add("SAVE", r, ao, an, f"all-correct both; mean saving {r['save']/1000:.1f}k")
# G2 GAIN: old 0/3, new solved; prefer new 2/2, then by new correct tokens asc? -> rank by #new correct desc, then old mean tok desc
g2 = [r for r in rows if not r["oc"] and r["nc"] and r["k"][0] not in used_tasks]
g2.sort(key=lambda r: (-len(r["nc"]), r["k"]))
for r in g2[:8]:
    ao, an = pick_attempt(r, lambda x: True, C)
    add("GAIN", r, ao, an, f"old 0/{len(r['o'])}, new {len(r['nc'])}/{len(r['n'])}")
# G3 WASTE: new correct attempt with >50k tokens; rank by that attempt's tokens desc
g3 = []
for r in rows:
    if r["k"][0] in used_tasks: continue
    for a in r["nc"]:
        if tok(r["n"][a]) > 50000: g3.append((tok(r["n"][a]), r, a))
g3.sort(key=lambda x: -x[0]); seen = set()
for t, r, a in g3:
    if r["k"][0] in seen: continue
    seen.add(r["k"][0])
    ao = a if a in r["o"] else 0
    add("WASTE", r, ao, a, f"new correct at {t/1000:.1f}k")
    if len(seen) == 6: break
# G4 REGRESS: new 0/2, old solved; rank by old #correct desc
g4 = [r for r in rows if not r["nc"] and r["oc"] and r["k"][0] not in used_tasks]
g4.sort(key=lambda r: (-len(r["oc"]), r["k"]))
for r in g4[:4]:
    ao, an = pick_attempt(r, C, lambda x: True)
    add("REGRESS", r, ao, an, f"old {len(r['oc'])}/{len(r['o'])}, new 0/{len(r['n'])}")

print("pool sizes: SAVE-eligible", len(g1), "GAIN", len(g2), "WASTE", len(seen), "of", len({x[1]['k'] for x in g3}), "REGRESS", len(g4))
for s in sel:
    print(s["group"], s["task"], s["pair"], "a%s/a%s" % (s["old_attempt"], s["new_attempt"]), s["old_tokens"], s["new_tokens"], s["old_pair_acc"], s["new_pair_acc"], s["why"])
json.dump(sel, open(Path(__file__).with_name("selection.json"), "w"), indent=1)
