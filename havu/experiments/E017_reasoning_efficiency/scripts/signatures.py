"""Population-level timeline signatures from the offline replay (all 835 sub80s40 + sub96n attempts). No API."""
import json, collections, statistics as st
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
idx = {(r["run"], r["task_id"], r["pair"], r["attempt"]): r for r in map(json.loads, open(OUT / "attempts_index.jsonl"))}
rep = {}
for fn in ("replay.jsonl", "replay_sample.jsonl"):
    for r in map(json.loads, open(OUT / fn)):
        rep[(r["run"], r["task_id"], r["pair"], r["attempt"])] = r
rows = []
for k, r in rep.items():
    tl = [t for t in r["timeline"] if t.get("probe")]
    tok = r["tokens"]
    first = lambda f: next((t["cum_tok"] for t in tl if f(t)), None)
    full = first(lambda t: t["full_train"])
    ok_state = first(lambda t: t["full_train_test_ok"] or t["answer_ok"] or t["any_cand_test_ok"])
    best = max((t["best_train"] / t["n_train"] for t in tl), default=0)
    last = tl[-1] if tl else {}
    rows.append(dict(key=k, outcome=r["outcome"], tokens=tok, first_full=full, first_ok=ok_state, best_frac=best,
                     final_full=bool(last.get("full_train")), final_ok=bool(last.get("answer_ok")),
                     fidelity=sum(t["same_output"] for t in r["timeline"]) / max(1, len(r["timeline"])),
                     ncells=len(r["timeline"]), grid_lines=idx[k]["grid_lines"], max_turn=idx[k]["max_turn_tok"],
                     turn1=idx[k]["turn_tok"][0] if idx[k]["turn_tok"] else 0, solved_any=idx[k]["pair_solved_any"],
                     first_tool_tok=idx[k]["turn_tok"][0] if idx[k]["turn_tok"] else 0))
res = {}
print("attempts", len(rows), "mean replay fidelity (cell output identical)", round(st.mean(r["fidelity"] for r in rows), 3))
for o in ("correct", "wrong", "none"):
    rs = [r for r in rows if r["outcome"] == o]
    ff = [r for r in rs if r["first_full"] is not None]
    print(f"\n{o}: n={len(rs)}  with probed train-passing fn at some point: {len(ff)} ({len(ff)/len(rs):.0%})")
    if ff:
        print("  first train-pass tok median", st.median(r["first_full"] for r in ff), " as % of attempt median", round(st.median(r["first_full"] / r["tokens"] for r in ff), 2),
              " tokens after first train-pass: mean", round(st.mean(r["tokens"] - r["first_full"] for r in ff)), "sum share of bucket", round(sum(r["tokens"] - r["first_full"] for r in ff) / sum(r["tokens"] for r in rs), 3))
    print("  best train fraction ever: mean", round(st.mean(r["best_frac"] for r in rs), 2), " never >0:", sum(r["best_frac"] == 0 for r in rs))
    print("  max single-turn tokens median", st.median(r["max_turn"] for r in rs), " turn-1 tokens median", st.median(r["turn1"] for r in rs), " grid-like lines median", st.median(r["grid_lines"] for r in rs))
    if o != "correct":
        had_ok = [r for r in rs if r["first_ok"] is not None]
        print("  attempts whose session held a correct test grid at some point but ended wrong:", len(had_ok))
# P(correct | no probed train-pass by X tokens), among attempts still running at X
print("\nEarly-stop table: among attempts still running at X tokens with no probed train-passing fn yet")
tab = []
for X in (10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000):
    alive = [r for r in rows if r["tokens"] > X and (r["first_full"] is None or r["first_full"] > X)]
    c = sum(r["outcome"] == "correct" for r in alive)
    rest = sum(r["tokens"] - X for r in alive)
    passed = [r for r in rows if r["tokens"] > X and r["first_full"] is not None and r["first_full"] <= X]
    cp = sum(r["outcome"] == "correct" for r in passed)
    tab.append((X, len(alive), c, rest))
    print(f"  X={X//1000}k: n={len(alive)} correct={c} ({c/max(1,len(alive)):.0%}) tokens spent after X={rest/1e6:.2f}M ({rest/sum(r['tokens'] for r in rows):.1%} of all) | already-passed&alive n={len(passed)} correct={cp/max(1,len(passed)):.0%}")
print("\nAmong attempts with a probed train-passing fn: P(correct)", round(sum(r['outcome']=='correct' for r in rows if r['first_full'] is not None)/sum(r['first_full'] is not None for r in rows),3),
      "; without:", round(sum(r['outcome']=='correct' for r in rows if r['first_full'] is None)/sum(r['first_full'] is None for r in rows),3))
# turn-1 length vs outcome
print("\nLongest single turn vs outcome")
for lo, hi in ((0, 10000), (10000, 20000), (20000, 30000), (30000, 99999)):
    rs = [r for r in rows if lo <= r["max_turn"] < hi]
    print(f"  max turn {lo//1000}-{hi//1000}k: n={len(rs)} correct {sum(r['outcome']=='correct' for r in rs)/max(1,len(rs)):.0%}")
for lo, hi in ((0, 1), (1, 10), (10, 40), (40, 9999)):
    rs = [r for r in rows if lo <= r["grid_lines"] < hi]
    print(f"  grid-like reasoning lines {lo}-{hi}: n={len(rs)} correct {sum(r['outcome']=='correct' for r in rs)/max(1,len(rs)):.0%} mean tok {st.mean(r['tokens'] for r in rs)/1e3:.1f}k")
json.dump([dict(r, key=list(r["key"])) for r in rows], open(OUT / "signatures.json", "w"))
