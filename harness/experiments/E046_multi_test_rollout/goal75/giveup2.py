"""Give-up rule with the trace signal: an E128 per-pair attempt with no all-train-pass output by T tokens is cut at T
(answer assumed wrong). Applied to per-pair records via (task, pair, attempt) from the dsv4-pruning analysis; multi-test
records are left untouched (no first-pass data), so savings on multi tasks are understated."""
import json, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds
rows = [r for r in json.load(open("/home/vu/Workspace/ARC-AGI-2/experiments/E043_dsv41_real_pruning/analysis/giving_up/results.json"))["rows"]
        if r["model"] in ("e128", "E128", "e128t")]
print("e128 rows:", len(rows), sorted({r["model"] for r in json.load(open("/home/vu/Workspace/ARC-AGI-2/experiments/E043_dsv41_real_pruning/analysis/giving_up/results.json"))["rows"]}))
fpt = {}
for r in rows:
    fpt.setdefault((r["key"][0], r["key"][1]), []).append(r)
# per-pair pool order matches records per (task, pair); match by (tokens) when possible
def cut(T):
    pp = {}
    lost = saved = tot = 0
    for k, recs in ORIG_PP.items():
        trace = fpt.get(k, [])
        out = []
        for r in recs:
            m = next((x for x in trace if abs(x["tokens"] - r["tokens"]) <= 2), None)
            r2 = dict(r)
            tot += r["tokens"]
            if m is not None and r["tokens"] > T and (m["first_pass_tok"] is None or m["first_pass_tok"] > T):
                lost += sum(r["correct"]); saved += r["tokens"] - T
                r2["tokens"] = T; r2["correct"] = [False] * len(r["correct"])
            out.append(r2)
        pp[k] = out
    return pp, lost, saved / tot

ORIG_PP = G.PP
matched = sum(1 for k, recs in G.PP.items() for r in recs if any(abs(x["tokens"] - r["tokens"]) <= 2 for x in fpt.get(k, [])))
print("per-pair records matched to traces:", matched, "of", sum(len(v) for v in G.PP.values()))
for T in (70000, 60000, 50000):
    if T:
        G.PP, lost, frac = cut(T); info = f"(per-pair records: {lost} correct lost, {100*frac:.1f}% of per-pair tokens saved)"
    else:
        G.PP, info = ORIG_PP, ""
    for label, kw in [("V6 now", {}), ("no thrash, no gate, x44", dict(thrash=False, gate=0, w=44)),
                      ("no thrash x44 + tokens x0.85", dict(thrash=False, gate=0, w=44, scale=0.85))]:
        a, b, c = G.run(**kw)
        print(f"T={T or '-':>6} {label:30s} score {a:5.1f}  a1% {c:5.1f} {info}")
G.PP = ORIG_PP
