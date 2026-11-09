"""Per-pair accuracy of finished attempts. Pairs matched to the local solutions by test-input content.
Usage: score.py <attempts.jsonl> <run challenges.json> <local challenges.json> <local solutions.json> [max_t_end_h]"""
import json, sys, collections
att, rch, lch, lsol = sys.argv[1:5]
tmax = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
rch, lch, lsol = (json.load(open(x)) for x in (rch, lch, lsol))
key = lambda g: json.dumps(g, separators=(",", ":"))
ref = {}
for t, v in lch.items():
    for i, x in enumerate(v["test"]):
        ref.setdefault(key(x["input"]), []).append(lsol[t][i])
recs = [json.loads(l) for l in open(att) if l.strip()]
recs = [r for r in recs if "task_id" in r and (r.get("t_end_h") or 0) <= tmax]
c = collections.Counter(); by_att = collections.Counter(); unmatched = 0; stops = collections.Counter()
for r in recs:
    t = r["task_id"]; n = r.get("multi_test")
    stops[r.get("stop_reason")] += 1
    if n:
        preds = r["pred"] if isinstance(r.get("pred"), list) and len(r["pred"]) == n else [None] * n
        items = [(q, preds[q]) for q in range(n)]
    else:
        items = [(r.get("pair", 0), r.get("pred"))]
    for q, p in items:
        sols = ref.get(key(rch[t]["test"][q]["input"]))
        if sols is None:
            unmatched += 1; continue
        ok = p is not None and p in sols
        c["pairs"] += 1; c["correct"] += ok; c["none"] += p is None
        by_att[(r["attempt"], "pairs")] += 1; by_att[(r["attempt"], "correct")] += ok
tot_out = sum(((r.get("usage") or {}).get("completion", 0)) % 10 ** 9 for r in recs)
print(f"records {len(recs)} (multi {sum(1 for r in recs if r.get('multi_test'))}), pair-preds {c['pairs']}, correct {c['correct']} "
      f"({100 * c['correct'] / max(1, c['pairs']):.1f}%), pred None {c['none']}, unmatched {unmatched}; output tokens {tot_out / 1e6:.2f}M")
for a in sorted({k[0] for k in by_att}):
    print(f"  attempt {a}: {by_att[(a, 'correct')]}/{by_att[(a, 'pairs')]}")
print("  stop reasons:", dict(stops))
