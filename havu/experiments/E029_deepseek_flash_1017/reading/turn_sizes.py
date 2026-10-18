"""Turn-size facts behind the strategy_reading.md proposals (all sub80s40 attempts, OLD vs NEW). No API calls.
Usage: .venv/bin/python -I turn_sizes.py"""
import gzip, json, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
F = {"OLD": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
     "NEW": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e029_flash1017_sub80s40.jsonl"}
for lab, f in F.items():
    A = []
    for l in open(f):
        r = json.loads(l)
        if "error" in r: continue
        d = json.load(gzip.open(r["trace_path"], "rt"))
        T = [e["usage"]["completion_tokens"] for e in d["events"] if "usage" in e]
        cum, cross, first_tool = 0, None, None
        for i, t in enumerate(T):
            if cum < 40000 <= cum + t: cross = (t, cum)   # turn during which 40k is crossed
            cum += t
        A.append(dict(c=r["correct"], tok=sum(T), T=T, cross=cross, turns=r["turns"], stop=r["stop_reason"]))
    tot = sum(a["tok"] for a in A)
    big = lambda k: sum(t for a in A for t in a["T"] if t >= k) / tot
    n = len(A)
    print(f"\n{lab}: {n} attempts, acc {sum(a['c'] for a in A)/n:.3f}, mean tok {tot/n/1e3:.1f}k, mean turns {st.mean(a['turns'] for a in A):.1f}")
    print(f"  share of output tokens in turns >=5k: {big(5000):.0%}, >=10k: {big(10000):.0%}, >=20k: {big(20000):.0%}")
    mx = [max(a["T"]) for a in A]
    print(f"  max single turn per attempt: median {st.median(mx)/1e3:.1f}k, p75 {sorted(mx)[3*n//4]/1e3:.1f}k; attempts with a >=15k turn: {sum(m>=15000 for m in mx)} ({sum(m>=15000 for m in mx)/n:.0%})")
    for lo, hi in ((0, 10000), (10000, 20000), (20000, 99999)):
        S = [a for a in A if lo <= max(a["T"]) < hi]
        if S: print(f"    max turn {lo/1e3:.0f}-{hi/1e3:.0f}k: n={len(S)} acc {sum(a['c'] for a in S)/len(S):.2f} mean tok {st.mean(a['tok'] for a in S)/1e3:.1f}k")
    X = [a for a in A if a["cross"]]
    inside = [a for a in X if a["cross"][0] >= 5000]
    print(f"  attempts passing 40k: {len(X)} ({len(X)/n:.0%}), acc {sum(a['c'] for a in X)/max(1,len(X)):.2f}; "
          f"40k crossed inside a >=5k turn (note arrives late): {len(inside)}, mean tokens of that turn past 40k "
          f"{st.mean(a['cross'][1]+a['cross'][0]-40000 for a in inside)/1e3 if inside else 0:.1f}k")
    print(f"  stop reasons: " + ", ".join(f"{k}={sum(a['stop']==k for a in A)}" for k in sorted({a['stop'] for a in A})))
    mt = [a for a in A if a["stop"] == "max_turns"]
    print(f"  max_turns attempts: tokens {[round(a['tok']/1e3,1) for a in mt]}")
    ge30 = [a for a in A if a["turns"] >= 30]
    print(f"  attempts with >=30 turns: {len(ge30)}, acc {sum(a['c'] for a in ge30)/max(1,len(ge30)):.2f}")
