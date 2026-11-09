"""Mechanical per-trace metrics for the 60 selected traces (cross-check for the close reading). No API calls.
Usage: .venv/bin/python -I trace_mech.py  -> prints a table, writes trace_mech.json next to this script.
'pass_tok' heuristic: cumulative tokens at the end of the first assistant turn whose tool output contains
>= n_train 'True' tokens and no 'False' (a train-check print), or 'all ... True'."""
import gzip, json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
rows = []
for s in json.load(open(HERE / "selection.json")):
    task = json.load(open(ROOT / f"data/arc-agi-2-github/data/evaluation/{s['task']}.json"))
    ntr = len(task["train"])
    for lab in ("old", "new"):
        d = json.load(gzip.open(s[f"{lab}_trace"], "rt"))
        usage = [e["usage"]["completion_tokens"] for e in d["events"] if "usage" in e]
        cum, ti, turns, pass_tok = 0, 0, [], None
        for m in d["messages"]:
            if m["role"] == "assistant":
                t = usage[ti] if ti < len(usage) else 0; cum += t; ti += 1
                turns.append(dict(t=t, ch=len(m.get("reasoning_content") or ""), cum=cum))
            elif m["role"] == "tool" and pass_tok is None:
                c = m["content"]
                if ("False" not in c and len(re.findall(r"\bTrue\b", c)) >= ntr) or re.search(r"all[^\n]{0,40}True", c):
                    pass_tok = cum
        tot = cum
        rows.append(dict(task=s["task"], pair=s["pair"], group=s["group"], model=lab.upper(),
            correct=s[f"{lab}_correct"], tokens=tot, turns=len(turns), turn1=turns[0]["t"] if turns else 0,
            max_turn=max(x["t"] for x in turns), big_turns=sum(x["t"] >= 5000 for x in turns),
            big_turn_tok=sum(x["t"] for x in turns if x["t"] >= 5000), pass_tok=pass_tok,
            after_pass=(tot - pass_tok) if pass_tok is not None else None,
            after40k=max(0, tot - 40000)))
json.dump(rows, open(HERE / "trace_mech.json", "w"), indent=0)
import statistics as st
for g in ("SAVE", "GAIN", "WASTE", "REGRESS", None):
    for mdl in ("OLD", "NEW"):
        R = [r for r in rows if r["model"] == mdl and (g is None or r["group"] == g)]
        med = lambda k: st.median([r[k] for r in R if r[k] is not None]) if any(r[k] is not None for r in R) else None
        mean = lambda k: st.mean([r[k] for r in R if r[k] is not None]) if any(r[k] is not None for r in R) else None
        print(f"{g or 'ALL':7s} {mdl} n={len(R):2d} tok {mean('tokens')/1e3:5.1f}k turns {mean('turns'):4.1f} turn1 med {med('turn1')/1e3:4.1f}k "
              f"maxturn mean {mean('max_turn')/1e3:5.1f}k big>=5k turns {mean('big_turns'):3.1f} ({mean('big_turn_tok')/1e3:4.1f}k) "
              f"pass_tok med {(med('pass_tok') or 0)/1e3:5.1f}k after_pass mean {(mean('after_pass') or 0)/1e3:4.1f}k n_pass {sum(r['pass_tok'] is not None for r in R)}")
