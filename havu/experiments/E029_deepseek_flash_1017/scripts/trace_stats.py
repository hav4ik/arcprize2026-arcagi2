"""E029: aggregate trace_features.jsonl into the tables of strategy_quant.md. No API calls.
Run: .venv/bin/python -I experiments/E029_deepseek_flash_1017/scripts/trace_stats.py [sub80s40|s20x60]
"""
import json, random, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = sys.argv[1] if len(sys.argv) > 1 else "sub80s40"
rows = [json.loads(l) for l in open(ROOT / "experiments/E029_deepseek_flash_1017/trace_features.jsonl")]
rows = [r for r in rows if r["cfg"] == CFG]
SUBMIT_FLOOR = 150  # tokens a bare submit turn costs (approx.; median submit-only turn, see below)


def q(x, p):
    x = sorted(v for v in x if v is not None)
    if not x:
        return float("nan")
    k = (len(x) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (k - lo)


def mean(x):
    x = [v for v in x if v is not None]
    return st.mean(x) if x else float("nan")


def med(x):
    return q(x, 0.5)


def derive(r):
    T = r["turns"]
    n = len(T)
    tok = [t["tok"] for t in T]
    # submit turn and answer-existence turn
    s = max((i for i, t in enumerate(T) if t["submit"]), default=None)
    a_print = next((i for i, t in enumerate(T) if t["has_pred"]), None)
    a_fb = None
    if r["has_pred"] and a_print is None:
        code = (T[s]["submit_code"] if s is not None else "answer") or "answer"
        m = re.match(r"\s*(\w+)", code)
        name = m.group(1) if m else "answer"
        end = s if s is not None else n
        a_fb = max((i for i in range(end) if name in T[i]["defs"]), default=max(0, end - 1))
    a = a_print if a_print is not None else a_fb
    d = next((i for i, t in enumerate(T) if t["has_def"]), None)
    end_h = a + 1 if a is not None else n
    start_h = max(1, d) if d is not None else end_h
    start_h = min(start_h, end_h)
    sol_first = next((i for i, t in enumerate(T) if t["has_sol"]), None)
    first_run = next((i for i, t in enumerate(T) if "run_python" in t["names"]), None)
    first_test = next((i for i, t in enumerate(T) if t["uses_test"]), None)
    tc_turns = [i for i, t in enumerate(T) if t["train_check"] is not None and (s is None or i < s)]
    first_allpass = next((i for i in tc_turns if T[i]["train_check"]["allpass"]), None)
    # last progress (for failed attempts): first appearance of eventual pred, or a new best train-check True count
    best, lp = -1, None
    for i in tc_turns:
        c = T[i]["train_check"]["true"]
        if c > best:
            best, lp = c, i
    lp = max([v for v in (lp, a, d) if v is not None], default=None)
    words = sum(t["words"] for t in T)
    tot = lambda k: sum(t[k] for t in T)
    ci_tails = [t["ci_tail_tok"] for t in T if t["ci_tail_tok"] is not None]
    hedge = {k: sum(t["hedge"][k] for t in T) for k in ["wait", "hmm", "actually", "maybe"]}
    g = {
        "tokens": sum(tok), "rtok": tot("rtok"), "vtok": tot("vtok"), "n_turns": n,
        "calls": tot("n_calls"), "tok_per_turn": sum(tok) / max(1, n),
        "turns_gt16k": sum(1 for x in tok if x > 16000), "turns_gt8k": sum(1 for x in tok if x > 8000),
        "max_turn": max(tok) if tok else 0,
        "ph_first": tok[0] if tok else 0, "ph_explore": sum(tok[1:start_h]), "ph_hyp": sum(tok[start_h:end_h]),
        "ph_tail": sum(tok[end_h:]) if a is not None else None,
        "ph_noans": sum(tok[end_h:]) if a is None else 0,
        "waste_err": sum(t["tok"] for t in T if t["err"]), "n_err_turns": sum(1 for t in T if t["err"]),
        "waste_len": sum(t["tok"] for t in T if t["finish"] == "length"),
        "a": a, "a_src": "print" if a_print is not None else ("fallback" if a_fb is not None else None), "d": d,
        "tail": sum(tok[a + 1:]) if a is not None else None,
        "tail_net": max(0, sum(tok[a + 1:]) - SUBMIT_FLOOR) if a is not None else None,
        "tail_turns": n - a - 1 if a is not None else None,
        "after_sol": sum(tok[sol_first + 1:]) if sol_first is not None else None,
        "sol_seen": sol_first is not None,
        "after_lp": sum(tok[lp + 1:]) if lp is not None else sum(tok),
        "tok_to_def": sum(tok[:d + 1]) if d is not None else None,
        "tok_to_first_run": sum(tok[:first_run + 1]) if first_run is not None else None,
        "turns_to_first_run": first_run + 1 if first_run is not None else None,
        "tok_to_first_test": sum(tok[:first_test + 1]) if first_test is not None else None,
        "turns_to_first_test": first_test + 1 if first_test is not None else None,
        "tok_to_answer": sum(tok[:a + 1]) if a is not None else None,
        "out_chars": tot("out_chars"), "code_chars": tot("code_chars"), "debug_print": tot("debug_print"),
        "n_print": tot("n_print"),
        "train_check": bool(tc_turns), "last_tc_allpass": bool(tc_turns) and T[tc_turns[-1]]["train_check"]["allpass"],
        "words": words,
        "pk_grid": tot("grid_lines") / max(1, words) * 1e3,
        "pk_coord": tot("coord_tuples") / max(1, words) * 1e3,
        "pk_intent": tot("code_intent") / max(1, words) * 1e3,
        "pk_reverify": tot("reverify") / max(1, words) * 1e3,
        "pk_mental": tot("mental") / max(1, words) * 1e3,
        "pk_giveup": tot("giveup") / max(1, words) * 1e3,
        "pk_reread_tool": tot("reread_tool") / max(1, words) * 1e3,
        "pk_reread_prompt": tot("reread_prompt") / max(1, words) * 1e3,
        "grid_lines": tot("grid_lines"), "reread_tool": tot("reread_tool"), "reread_prompt": tot("reread_prompt"),
        "coord_own_tok": sum(t["coord_own_chars"] / (t["rchars"] / t["rtok"]) if t["rtok"] and t["rchars"] else 0
                             for t in T),
        "ci_tail_mean": mean(ci_tails) if ci_tails else None, "ci_tail_sum": sum(ci_tails),
        "ci_tail_gt2k": sum(1 for x in ci_tails if x > 2000),
        "chars_per_tok": sum(t["rchars"] for t in T) / max(1, tot("rtok")),
    }
    for k, v in hedge.items():
        g["pk_" + k] = v / max(1, words) * 1e3
    g["pk_hedge4"] = sum(hedge.values()) / max(1, words) * 1e3
    # what-ifs
    rt = [t["rtok"] for t in T]
    for cap in (8000, 12000, 16000, 24000):
        g[f"cap{cap // 1000}_save"] = sum(max(0, x - cap) for x in rt)
        g[f"cap{cap // 1000}_risk"] = any(rt[i] > cap for i in range(0, (a + 1) if a is not None else n))
    for cap in (2000, 4000, 8000):
        g[f"t1cap{cap // 1000}_save"] = max(0, rt[0] - cap) if rt else 0
        g[f"t1cap{cap // 1000}_risk"] = bool(rt) and rt[0] > cap
    # implementable stop rule: after a train check printed all-pass, stop at the first test grid produced
    # by a function call on test_input (that turn or later) and submit that grid.
    rule = {}
    for name, need_tc in (("A", True), ("B", False)):
        hit = None
        for i, t in enumerate(T):
            if s is not None and i >= s:
                break
            if need_tc and (first_allpass is None or i < first_allpass):
                continue
            if not need_tc and (d is None or i < d):
                continue
            if t["test_grids"]:
                hit = (i, t["test_grids"][0][0], any(g[0] for g in t["test_grids"]), any(g[1] for g in t["test_grids"]))
                break
        if hit:
            rule[name] = {"turn": hit[0], "correct": hit[1], "any_correct": hit[2], "any_pred": hit[3], "save": max(0, sum(tok[hit[0] + 1:]) - SUBMIT_FLOOR)}
        else:
            rule[name] = None
    g["rule"] = rule
    return g


for r in rows:
    r.update(derive(r))

by = defaultdict(list)
for r in rows:
    by[r["run"]].append(r)
OLD, NEW = by["old"], by["new"]


def pairmean(R, key, filt):
    d = defaultdict(list)
    for r in R:
        if filt(r) and r[key] is not None:
            d[(r["task_id"], r["pair"])].append(r[key])
    return {k: mean(v) for k, v in d.items()}


def paired(key, filt=lambda r: r["correct"], B=2000):
    o, n = pairmean(OLD, key, filt), pairmean(NEW, key, filt)
    ks = sorted(set(o) & set(n))
    diffs = [n[k] - o[k] for k in ks]
    tasks = sorted({k[0] for k in ks})
    bt = defaultdict(list)
    for k, dv in zip(ks, diffs):
        bt[k[0]].append(dv)
    rng = random.Random(0)
    boots = []
    for _ in range(B):
        samp = [rng.choice(tasks) for _ in tasks]
        vals = [v for t in samp for v in bt[t]]
        boots.append(mean(vals))
    boots.sort()
    return {"n": len(ks), "old_mean": mean([o[k] for k in ks]), "new_mean": mean([n[k] for k in ks]),
            "old_med": med([o[k] for k in ks]), "new_med": med([n[k] for k in ks]),
            "d_mean": mean(diffs), "d_med": med(diffs), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)]}


def fk(v, k=True):
    if v is None or v != v:
        return "–"
    return f"{v / 1e3:.1f}k" if k else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")


def table(title, metrics, kilo):
    print(f"\n### {title}")
    print("| Metric | old all | new all | old correct | new correct | old failed | new failed | paired both-correct old → new (Δ mean [95% CI]; Δ median) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, lab in metrics:
        k = kilo(key)
        cells = []
        for grp in (lambda r: True, lambda r: r["correct"], lambda r: not r["correct"]):
            for R in (OLD, NEW):
                x = [r[key] for r in R if grp(r) and r[key] is not None]
                cells.append(f"{fk(mean(x), k)} / {fk(med(x), k)}")
        p = paired(key)
        cells.append(f"{fk(p['old_mean'], k)} → {fk(p['new_mean'], k)} ({fk(p['d_mean'], k)} [{fk(p['lo'], k)}, {fk(p['hi'], k)}]; "
                     f"{fk(p['d_med'], k)})")
        print(f"| {lab} | " + " | ".join(cells) + " |")


print(f"# Config {CFG}: old n={len(OLD)} acc={mean([r['correct'] for r in OLD]):.1%}; new n={len(NEW)} "
      f"acc={mean([r['correct'] for r in NEW]):.1%}")
p = paired("tokens")
print(f"Paired both-correct pairs: {p['n']}")
print("cells: mean / median")

KTOK = lambda key: key not in {"n_turns", "calls", "turns_gt16k", "turns_gt8k", "n_err_turns", "tail_turns",
                                "turns_to_first_run", "turns_to_first_test", "debug_print", "n_print", "chars_per_tok"} \
    and not key.startswith("pk_")
table("Tokens by phase (output tokens per attempt)", [
    ("tokens", "Total output tokens"), ("rtok", "Reasoning (thinking) tokens"), ("vtok", "Visible tokens (content + tool-call code)"),
    ("ph_first", "Phase 1: first turn (before first tool call)"), ("ph_explore", "Phase 2: exploration (to first def solve)"),
    ("ph_hyp", "Phase 3: hypothesis / debug (def solve → submitted grid exists)"),
    ("ph_tail", "Phase 4: verification tail (after submitted grid exists; attempts with an answer)"),
    ("ph_noans", "No-answer remainder (attempts with no submitted grid)"),
    ("waste_err", "Overlay: tokens in turns whose code errored"),
    ("waste_len", "Overlay: tokens in truncated (length) turns"),
], KTOK)
table("Turn structure", [
    ("n_turns", "Turns"), ("calls", "Tool calls"), ("tok_per_turn", "Tokens per turn"), ("max_turn", "Largest turn"),
    ("turns_gt8k", "Turns > 8k tokens"), ("turns_gt16k", "Turns > 16k tokens"), ("n_err_turns", "Turns with code errors"),
    ("chars_per_tok", "Reasoning chars per token"),
], KTOK)
table("Strategy markers per 1k reasoning words", [
    ("words", "Reasoning words per attempt"), ("pk_grid", "Hand-transcribed grid rows"),
    ("pk_reread_tool", "  of which rows copied from earlier tool output"), ("pk_reread_prompt", "  of which rows copied from the prompt"),
    ("pk_coord", "Coordinate tuples (r,c) in prose"), ("coord_own_tok", "Est. tokens in cell-coordinate lines (E026 def.)"),
    ("pk_intent", "'Let me check this in code/python' statements (explicit tool intent)"),
    ("pk_mental", "'Let's compute/count/check/trace' with no code mention (mental computation)"),
    ("ci_tail_mean", "Prose tokens after such a statement, same turn (mean per statement-turn)"),
    ("ci_tail_sum", "Prose tokens after such statements, per attempt"),
    ("pk_wait", "'wait'"), ("pk_hmm", "'hmm'"), ("pk_actually", "'actually'"), ("pk_maybe", "'maybe'"),
    ("pk_hedge4", "Hedges total (wait+hmm+actually+maybe)"),
    ("pk_reverify", "Re-verification phrases"), ("pk_giveup", "Give-up phrases ('given the time', 'best guess')"),
], KTOK)
table("Code use", [
    ("tok_to_first_run", "Tokens to first code run (incl. that turn)"), ("turns_to_first_run", "Turn of first code run"),
    ("tok_to_first_test", "Tokens to first code touching test_input"), ("turns_to_first_test", "Turn of first test_input code"),
    ("tok_to_def", "Tokens to first def solve (incl. that turn)"), ("tok_to_answer", "Tokens until submitted grid exists"),
    ("calls", "Tool calls"), ("out_chars", "Tool output chars (excl. budget notes)"), ("code_chars", "Code chars written"),
    ("n_print", "print() calls in code"), ("debug_print", "Debug prints (mismatch/diff/where)"),
], lambda k: k in {"tok_to_first_run", "tok_to_first_test", "tok_to_def", "tok_to_answer", "out_chars", "code_chars"})

print("\n### Train check before submit (share of attempts)")
print("| | old all | new all | old correct | new correct | old failed | new failed |\n|---|---|---|---|---|---|---|")
for key, lab in [("train_check", "Has a code cell comparing solve() to train outputs before submitting"),
                 ("last_tc_allpass", "Last such check printed all-pass"), ("sol_seen", "True solution grid appeared in some tool output")]:
    cells = []
    for grp in (lambda r: True, lambda r: r["correct"], lambda r: not r["correct"]):
        for R in (OLD, NEW):
            cells.append(f"{mean([r[key] for r in R if grp(r)]):.0%}")
    print(f"| {lab} | " + " | ".join(cells) + " |")

# ---------- decomposition of the token drop ----------
print("\n### Decomposition of the mean token drop (all attempts)")
po, pn = mean([r["correct"] for r in OLD]), mean([r["correct"] for r in NEW])
tco, tcn = mean([r["tokens"] for r in OLD if r["correct"]]), mean([r["tokens"] for r in NEW if r["correct"]])
tfo, tfn = mean([r["tokens"] for r in OLD if not r["correct"]]), mean([r["tokens"] for r in NEW if not r["correct"]])
tot_d = mean([r["tokens"] for r in NEW]) - mean([r["tokens"] for r in OLD])
print(f"total Δ {tot_d/1e3:.2f}k; within-correct {pn*(tcn-tco)/1e3:.2f}k; within-failed {(1-pn)*(tfn-tfo)/1e3:.2f}k; "
      f"mix (more correct attempts) {(pn-po)*(tco-tfo)/1e3:.2f}k")
print("| Phase | old mean | new mean | Δ | share of total Δ |\n|---|---|---|---|---|")
for k in ["ph_first", "ph_explore", "ph_hyp", "ph_tail", "ph_noans"]:
    o = mean([r[k] or 0 for r in OLD]); nn = mean([r[k] or 0 for r in NEW])
    print(f"| {k} | {o/1e3:.2f}k | {nn/1e3:.2f}k | {(nn-o)/1e3:+.2f}k | {(nn-o)/tot_d:.0%} |")
print("Same, paired both-correct pairs:")
print("| Phase | old mean | new mean | Δ [95% CI] | share |\n|---|---|---|---|---|")
pt = paired("tokens")
for k in ["ph_first", "ph_explore", "ph_hyp", "ph_tail", "rtok", "vtok"]:
    p = paired(k)
    print(f"| {k} | {p['old_mean']/1e3:.2f}k | {p['new_mean']/1e3:.2f}k | {p['d_mean']/1e3:+.2f}k [{p['lo']/1e3:+.2f}, {p['hi']/1e3:+.2f}] | {p['d_mean']/pt['d_mean']:.0%} |")

# ---------- remaining waste in NEW ----------
for lab, R in (("new", NEW), ("old", OLD)):
    alltok = sum(r["tokens"] for r in R)
    C = [r for r in R if r["correct"]]
    F = [r for r in R if not r["correct"]]
    t = [r["tail"] for r in C]
    print(f"\n### Tail after the submitted grid first exists ({lab}, correct attempts n={len(C)})")
    print(f"answer located by print: {mean([r['a_src']=='print' for r in C]):.0%}, fallback (def/assign of submitted name): "
          f"{mean([r['a_src']=='fallback' for r in C]):.0%}")
    print(f"tail tokens: mean {mean(t)/1e3:.2f}k median {med(t)/1e3:.2f}k p75 {q(t,.75)/1e3:.2f}k p95 {q(t,.95)/1e3:.2f}k; "
          f"tail turns median {med([r['tail_turns'] for r in C])}; net of {SUBMIT_FLOOR}-tok submit: "
          f"sum {sum(r['tail_net'] for r in C)/1e3:.0f}k = {sum(r['tail_net'] for r in C)/alltok:.1%} of all {lab} tokens, "
          f"{sum(r['tail_net'] for r in C)/sum(r['tokens'] for r in C):.1%} of correct-attempt tokens")
    print(f"tail share per attempt: median {med([r['tail']/r['tokens'] for r in C]):.1%}; attempts with tail >5k: "
          f"{mean([r['tail']>5000 for r in C]):.0%}, >10k: {mean([r['tail']>10000 for r in C]):.0%}")
    ta = [r["tail"] for r in F if r["tail"] is not None]
    print(f"failed attempts n={len(F)} (with answer {len(ta)}): tail after their submitted grid mean {mean(ta)/1e3:.2f}k median {med(ta)/1e3:.2f}k; "
          f"tokens after last progress mean {mean([r['after_lp'] for r in F])/1e3:.2f}k median {med([r['after_lp'] for r in F])/1e3:.2f}k "
          f"p75 {q([r['after_lp'] for r in F],.75)/1e3:.2f}k p95 {q([r['after_lp'] for r in F],.95)/1e3:.2f}k; "
          f"sum {sum(r['after_lp'] for r in F)/1e3:.0f}k = {sum(r['after_lp'] for r in F)/alltok:.1%} of all tokens")
    print(f"failed attempts where the true solution appeared in a tool output: {sum(r['sol_seen'] for r in F)}")
    print(f"buckets (share of all {lab} tokens): ", end="")
    buck = {
        "correct: first turn": sum(r["ph_first"] for r in C), "correct: explore": sum(r["ph_explore"] for r in C),
        "correct: hyp/debug": sum(r["ph_hyp"] for r in C), "correct: tail": sum(r["tail"] for r in C),
        "failed: before last progress": sum(r["tokens"] - r["after_lp"] for r in F),
        "failed: after last progress": sum(r["after_lp"] for r in F),
    }
    print("; ".join(f"{k} {v/alltok:.1%}" for k, v in buck.items()))
    # tail contents
    print("tail composition (correct): turns in tail with code errors / no code / test re-runs not measured; "
          f"tail reasoning share {sum(sum(t2['rtok'] for t2 in r['turns'][r['a']+1:]) for r in C)/max(1,sum(r['tail'] for r in C)):.0%}")

# ---------- what-ifs ----------
print("\n### What-ifs (NEW, offline; savings are upper bounds as a cut turn may be followed by others)")
alltok = sum(r["tokens"] for r in NEW)
C = [r for r in NEW if r["correct"]]
print(f"total NEW tokens {alltok/1e3:.0f}k over {len(NEW)} attempts; correct {len(C)}")
print("| Rule | tokens saved (share of all) | correct attempts at risk (cut before answer existed) | failed attempts affected |\n|---|---|---|---|")
sv = sum(r["tail_net"] or 0 for r in NEW)
print(f"| Oracle: stop when eventually-submitted grid first exists | {sv/1e3:.0f}k ({sv/alltok:.1%}) | 0 (by construction) | same grid |")
for cap in (8, 12, 16, 24):
    sv = sum(r[f"cap{cap}_save"] for r in NEW)
    print(f"| Per-turn thinking cap {cap}k | {sv/1e3:.0f}k ({sv/alltok:.1%}) | {sum(r[f'cap{cap}_risk'] for r in C)} of {len(C)} | "
          f"{sum(r[f'cap{cap}_risk'] for r in NEW if not r['correct'])} |")
for cap in (2, 4, 8):
    sv = sum(r[f"t1cap{cap}_save"] for r in NEW)
    print(f"| First-turn thinking cap {cap}k | {sv/1e3:.0f}k ({sv/alltok:.1%}) | {sum(r[f't1cap{cap}_risk'] for r in C)} of {len(C)} | "
          f"{sum(r[f't1cap{cap}_risk'] for r in NEW if not r['correct'])} |")
for name, lab in (("A", "Stop at first test grid after an all-pass train check"),
                  ("B", "Stop at first test grid after first def solve (no train check)")):
    trig = [r for r in NEW if r["rule"][name]]
    sv = sum(r["rule"][name]["save"] for r in trig)
    lost = sum(1 for r in trig if r["correct"] and not r["rule"][name]["correct"])
    lost_opt = sum(1 for r in trig if r["correct"] and not r["rule"][name]["any_correct"])
    gained = sum(1 for r in trig if not r["correct"] and r["rule"][name]["any_correct"])
    print(f"| {lab} | {sv/1e3:.0f}k ({sv/alltok:.1%}); fires on {len(trig)}/{len(NEW)} | {lost_opt}–{lost} correct → wrong "
          f"(optimistic: any grid printed that turn; pessimistic: first grid) | {gained} wrong → correct |")

# per-turn distributions
print("\n### Per-turn thinking distribution (NEW vs OLD)")
for lab, R in (("old", OLD), ("new", NEW)):
    rt = [t["rtok"] for r in R for t in r["turns"]]
    big = sum(x for x in rt if x > 16000)
    print(f"{lab}: turns {len(rt)}, p50 {q(rt,.5):.0f} p90 {q(rt,.9):.0f} p99 {q(rt,.99):.0f}; tokens in turns >16k thinking: "
          f"{big/sum(r['tokens'] for r in R):.1%}; first-turn thinking median {med([r['turns'][0]['rtok'] for r in R]):.0f} "
          f"mean {mean([r['turns'][0]['rtok'] for r in R]):.0f} p90 {q([r['turns'][0]['rtok'] for r in R],.9):.0f}")
# submit-only turn cost (calibrates SUBMIT_FLOOR)
st_ = [r["turns"][-1]["tok"] for r in NEW if r["turns"] and r["turns"][-1]["submit"]]
print(f"last (submit) turn tokens NEW: median {med(st_):.0f} mean {mean(st_):.0f}")

print("\n### What-if: hard total-output cap (NEW); correct attempts whose submitted grid appeared after the cap are lost")
print("| Cap | tokens saved (share) | correct attempts lost | of which pairs where the other attempt still correct |\n|---|---|---|---|")
for cap in (30000, 40000, 50000, 60000):
    sv = sum(max(0, r["tokens"] - cap) for r in NEW)
    lost = [r for r in C if (r["tok_to_answer"] or 0) > cap]
    print(f"| {cap//1000}k | {sv/1e3:.0f}k ({sv/alltok:.1%}) | {len(lost)} of {len(C)} | – |")
print("\n### What-if: abandon if no all-pass train check by X tokens (NEW)")
for X in (20000, 30000, 40000):
    sv, lost = 0, 0
    for r in NEW:
        T = r["turns"]; cum = 0; fa = None
        for i, t in enumerate(T):
            cum += t["tok"]
            if t["train_check"] and t["train_check"]["allpass"]:
                fa = cum; break
        if (fa is None or fa > X) and r["tokens"] > X:
            sv += r["tokens"] - X
            lost += r["correct"]
    print(f"X={X//1000}k: saves {sv/1e3:.0f}k ({sv/alltok:.1%}); correct attempts lost {lost} of {len(C)}")

print("\n### Big turns and deferred-code prose")
for lab, R in (("old", OLD), ("new", NEW)):
    allt = sum(r["tokens"] for r in R)
    TT = [t for r in R for t in r["turns"]]
    big = [t for t in TT if t["rtok"] > 8000]
    bt = sum(t["tok"] for t in big)
    ci = [t for t in TT if t["ci_tail_tok"] is not None]
    cit = sum(t["ci_tail_tok"] for t in ci)
    for cut in (500, 1000, 2000):
        pass
    sv = {cut: sum(max(0, t["ci_tail_tok"] - cut) for t in ci) for cut in (500, 1000, 2000)}
    coord_big = sum(t["coord_own_chars"] + t["grid_chars"] for t in big) / max(1, sum(t["rchars"] for t in big))
    coord_small = sum(t["coord_own_chars"] + t["grid_chars"] for t in TT if t["rtok"] <= 8000) / max(1, sum(t["rchars"] for t in TT if t["rtok"] <= 8000))
    big_ci = sum(1 for t in big if t["ci_tail_tok"] is not None)
    hed_big = sum(sum(t["hedge"].values()) for t in big) / max(1, sum(t["words"] for t in big)) * 1e3
    hed_small = sum(sum(t["hedge"].values()) for t in TT if t["rtok"] <= 8000) / max(1, sum(t["words"] for t in TT if t["rtok"] <= 8000)) * 1e3
    print(f"{lab}: turns >8k thinking: {len(big)} ({len(big)/len(TT):.1%} of turns) hold {bt/allt:.1%} of tokens; "
          f"{big_ci/max(1,len(big)):.0%} of them contain a 'check in code' statement; cell-coordinate+grid-row share of chars "
          f"big {coord_big:.1%} vs small {coord_small:.1%}; hedges/1k words big {hed_big:.1f} vs small {hed_small:.1f}")
    print(f"   prose after a 'check in code' statement (same turn): {cit/allt:.1%} of all tokens; "
          f"trimming each to ≤0.5k/1k/2k would save {sv[500]/allt:.1%}/{sv[1000]/allt:.1%}/{sv[2000]/allt:.1%} (upper bound)")
