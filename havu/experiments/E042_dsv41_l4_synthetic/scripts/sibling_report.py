#!/usr/bin/env python3
"""E042: summarize sibling_transfer.py rows against the pairs' own attempts.
Usage: sibling_report.py <rows.jsonl> <results.jsonl>"""
import collections
import json
import sys

rows = [r for r in map(json.loads, open(sys.argv[1])) if "error" not in r]
recs = [r for r in map(json.loads, open(sys.argv[2])) if "task_id" in r]
own = collections.defaultdict(list)
for r in recs:
    own[(r["task_id"], r["pair"])].append(r)
print(f"{len(rows)} attempts replayed")

c = collections.Counter()
for r in rows:
    for q, ok in r["sib"].items():
        c[(r["train_pass"] is True, r["correct"], ok)] += 1
print("source attempt: train_pass, correct -> sibling answer correct: count")
for k, v in sorted(c.items()):
    print(f"  tp={k[0]!s:5s} correct={k[1]!s:5s} -> sibling {k[2]!s:5s}: {v}")

# Per sibling pair: transfers from train-passing attempts of the other pairs
xfer = collections.defaultdict(list)
for r in rows:
    if r["train_pass"] is True:
        for q, ok in r["sib"].items():
            xfer[(r["task_id"], int(q))].append(ok)
multi = sorted({k for k in own if any(k2[0] == k[0] and k2 != k for k2 in own)})
t = collections.Counter()
for k in multi:
    o = any(r["correct"] for r in own[k])
    x = xfer.get(k, [])
    t["pairs"] += 1
    t["own_pass2"] += o
    if x:
        t["with_xfer"] += 1
        t["xfer_first_ok"] += x[0]
        t["xfer_any_ok"] += any(x)
        t["own_ok_when_xfer"] += o
        t["union_when_xfer"] += o or any(x)
        t["own_tokens_when_xfer"] += sum((r.get("usage") or {}).get("completion") or 0 for r in own[k])
    t["union"] += o or any(x)
t["all_tokens"] = sum((r.get("usage") or {}).get("completion") or 0 for r in recs)
print(dict(t))
print(f"multi-pair pairs {t['pairs']}: own pass@2 {t['own_pass2']}, own-or-transfer {t['union']}; "
      f"{t['with_xfer']} have a train-passing transfer, its first answer right {t['xfer_first_ok']} "
      f"(own pass@2 on those {t['own_ok_when_xfer']}); their own attempts' tokens "
      f"{t['own_tokens_when_xfer'] / t['all_tokens']:.1%} of all")

# Answer selection with transfers (needs sib_hash rows). own: the pair's attempts ranked as ranked_answers();
# A = own only; B = own train-passing, then transfers from train-passing siblings, then own others; C = own, then
# transfers into a free slot; D = no own attempts at all (deadline), transfers only.
if rows and "sib_hash" in rows[0]:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "evals"))
    from train_check import grid_hash
    xh = collections.defaultdict(list)
    for r in rows:
        if r["train_pass"] is True:
            for q, h in r["sib_hash"].items():
                if not h.startswith("ERR"):
                    xh[(r["task_id"], int(q))].append((h, r["sib"][q]))

    def top2(cands):
        out = []
        for h, ok in cands:
            if h not in [o[0] for o in out]:
                out.append((h, ok))
            if len(out) == 2:
                break
        return any(ok for _, ok in out)

    s = collections.Counter()
    for k in own:
        o = sorted((r for r in own[k] if r.get("pred") is not None),
                   key=lambda r: (r.get("train_pass") is not True, r["attempt"]))
        oc = [(grid_hash(r["pred"]), bool(r["correct"])) for r in o]
        otp = [c for c, r in zip(oc, o) if r.get("train_pass") is True]
        orest = [c for c, r in zip(oc, o) if r.get("train_pass") is not True]
        x = xh.get(k, [])
        s["A"] += top2(oc)
        s["B"] += top2(otp + x + orest)
        s["C"] += top2(oc + x)
        s["D"] += top2(x)
        s["n"] += 1
    print(f"all {s['n']} pairs: own only {s['A']}, tp-own > transfer > other-own {s['B']}, own then transfer {s['C']}; "
          f"transfers alone {s['D']}")
