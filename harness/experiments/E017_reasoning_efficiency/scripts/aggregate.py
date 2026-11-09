"""Merge annotations/R*.jsonl into annotations.jsonl and print bucket-level summaries."""
import json, glob, collections, statistics as st
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
sample = {s["sid"]: s for s in map(json.loads, open(OUT / "sample.jsonl"))}
ann = {}
for f in sorted(glob.glob(str(OUT / "annotations/R*.jsonl"))):
    for l in open(f):
        if l.strip():
            a = json.loads(l); a["reader"] = Path(f).stem; ann[a["sid"]] = a
with open(OUT / "annotations.jsonl", "w") as f:
    for sid in sorted(ann):
        a = ann[sid]; a["bucket"] = sample[sid]["bucket"]; f.write(json.dumps(a) + "\n")
print("annotated", len(ann), "missing", sorted(set(sample) - set(ann)))
groups = {"wrong_solvable": [], "never_solved": [], "correct_long": [], "correct_short": []}
for a in ann.values():
    groups[a["bucket"]].append(a)
cats = ["perception", "hypothesis_search", "code_writing", "debugging", "verification", "post_success", "other"]
def num(x):
    try: return float(x)
    except Exception: return None
for g, rs in groups.items():
    if not rs: continue
    toks = [sample[a["sid"]]["tokens"] for a in rs]
    print(f"\n== {g} n={len(rs)} tokens mean {st.mean(toks)/1e3:.1f}k")
    # token-weighted shares
    sh = {c: sum(a["token_share"].get(c, 0) * sample[a["sid"]]["tokens"] for a in rs) / sum(toks) for c in cats}
    print("  token share (token-weighted %):", {c: round(v, 1) for c, v in sh.items()})
    print("  wasted % (token-weighted):", round(sum((num(a.get("wasted_tokens_pct")) or 0) * sample[a["sid"]]["tokens"] for a in rs) / sum(toks), 1))
    for fld in ["failure_mode", "best_intervention", "second_intervention", "checked_all_train_before_submit", "train_check_would_catch", "test_input_examined", "test_needs_beyond_train"]:
        print(f"  {fld}:", dict(collections.Counter(str(a.get(fld)) for a in rs).most_common()))
    print("  contradiction_ignored:", sum(bool(a.get("contradiction_ignored")) for a in rs), "perception errors >0:", sum(bool(a.get("perception_errors")) for a in rs))
    print("  n_hypotheses median:", st.median(num(a.get("n_hypotheses")) or 0 for a in rs))
    for fld in ["correct_idea_first_pct", "correct_rule_complete_pct", "first_train_pass_pct"]:
        v = [num(a.get(fld)) for a in rs]; vv = [x for x in v if x is not None]
        print(f"  {fld}: present {len(vv)}/{len(v)}", f"median {st.median(vv):.0f}" if vv else "")
