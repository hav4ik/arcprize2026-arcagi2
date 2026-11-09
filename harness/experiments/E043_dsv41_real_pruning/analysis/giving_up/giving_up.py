#!/usr/bin/env python3
"""E043: do pruned models give up, guess or stall? Tool-harness traces, read-only (parsing from ../doubt/common.py).

Per model (full_api, full_sh, E128, E112, E96), attempt level:
- how attempts end: submit / 40-turn cap / 80K budget; submits with no all-train-pass output in the attempt
  ("unverified"), split by whether they come before or after the 40K soft target, and their accuracy;
- trivial answers (the test input unchanged, or a single colour);
- giving-up language in thinking (best guess / just submit / give up / cannot find ...), and where it appears;
- stalling: turns and tokens before the first train check, the longest run of turns without a check, and how much
  of the attempt runs after the last turn that produced new evidence (a check, a new verdict);
- effort in the last turns of attempts that never pass (thinking per turn vs the attempt's own mean).
Writes results.json next to this script.
"""
import collections
import json
import os
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "doubt"))
from common import attempt_turns, is_train_check, load_records, load_trace, verdict  # noqa: E402

MODELS = ["full_api", "full_sh", "E128", "E112", "E96"]
TESTS = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_challenges.json"))
GIVEUP = re.compile(r"\b(?:give up|giving up|best guess|(?:just|simply) (?:submit|go with|guess|output|return)|"
                    r"submit (?:anyway|something|what we have|the current|our current|current|best)|"
                    r"can(?:no|')t (?:figure|find|determine|solve|deduce|get)|unable to (?:find|determine|solve)|"
                    r"no (?:clear|obvious) (?:rule|pattern)|good enough|settle for|take a guess|"
                    r"(?:as a|a) fallback|run(?:ning)? out of|out of (?:time|budget|tokens))\b", re.I)
BUDGET_TALK = re.compile(r"\b(?:budget|token(?:s)? (?:used|left|limit)|limit is|hard limit|past the target|"
                         r"time is (?:short|limited)|running low|over budget)\b", re.I)


def analyze(args):
    model, key, rec = args
    try:
        turns, ntr = attempt_turns(load_trace(rec["trace_path"]))
    except Exception as e:
        return {"err": repr(e)}
    tok = 0
    first_check = first_pass = None
    checks, last_new = [], -1
    streak = best_streak = 0
    giveup = []
    budget_talk = 0
    seen_code = set()
    for i, t in enumerate(turns):
        tok += t["comp"] or 0
        t["cum"] = tok
        v = verdict(t["code"], t["out"], ntr)
        chk = bool(t["code"]) and is_train_check(t["code"])
        if chk and first_check is None:
            first_check = i
        if v == "pass" and first_pass is None:
            first_pass = i
        norm = re.sub(r"\s+", "", t["code"])
        new = (chk or v is not None) or (norm and norm not in seen_code and not t["submit"])
        seen_code.add(norm)
        if chk:
            checks.append(v)
            last_new = i
        streak = 0 if chk else streak + 1
        best_streak = max(best_streak, streak)
        for m in GIVEUP.finditer(t["think"] + "\n" + t["content"]):
            giveup.append((i, tok, m.group(0).lower()))
        budget_talk += len(BUDGET_TALK.findall(t["think"]))
    total = tok or 1
    sub_turn = next((i for i, t in enumerate(turns) if t["submit"]), None)
    stop = rec.get("stop_reason")
    pred = rec.get("pred")
    test_in = TESTS.get(rec["task_id"], {}).get("test", [{}] * 9)[int(rec["pair"])].get("input")
    flat = [c for row in (pred or []) for c in (row if isinstance(row, list) else [row])]
    trivial = "identity" if pred is not None and pred == test_in else (
        "one_colour" if flat and len(set(flat)) == 1 else None)
    think_tok = [len(t["think"]) for t in turns]
    last = think_tok[-3:]
    return {
        "model": model, "key": key, "correct": bool(rec.get("correct")), "stop": stop, "tokens": tok,
        "turns": len(turns), "submitted": stop == "submitted", "train_pass": rec.get("train_pass"),
        "any_pass": first_pass is not None, "submit_tok": turns[sub_turn]["cum"] if sub_turn is not None else None,
        "first_check_turn": first_check, "first_check_tok": turns[first_check]["cum"] if first_check is not None else None,
        "first_pass_turn": first_pass, "first_pass_tok": turns[first_pass]["cum"] if first_pass is not None else None, "n_checks": len(checks), "n_fail_checks": sum(c == "fail" for c in checks),
        "best_streak": best_streak, "tail_after_last_check": (total - turns[last_new]["cum"]) / total if last_new >= 0 else 1.0,
        "giveup": giveup, "budget_talk": budget_talk, "trivial": trivial,
        "last3_vs_mean_think": (statistics.mean(last) / max(1, statistics.mean(think_tok))) if len(think_tok) >= 6 else None,
    }


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main():
    jobs = []
    for m in MODELS:
        for k, r in load_records(m).items():
            if r.get("trace_path") and os.path.exists(r["trace_path"]):
                jobs.append((m, k, r))
    with ProcessPoolExecutor(20) as pool:
        rows = [r for r in pool.map(analyze, jobs, chunksize=4) if "err" not in r]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    res = {}
    for m in MODELS:
        R = by[m]
        n = len(R)
        sub = [r for r in R if r["submitted"]]
        unver = [r for r in sub if not r["any_pass"]]
        unver_early = [r for r in unver if (r["submit_tok"] or 0) < 40000]
        never = [r for r in R if not r["any_pass"]]
        gu = [r for r in R if r["giveup"]]
        gu_early = [r for r in R if any(tk < 40000 for _, tk, _ in r["giveup"])]
        fc = [r["first_check_tok"] for r in R if r["first_check_tok"] is not None]
        fct = [r["first_check_turn"] for r in R if r["first_check_turn"] is not None]
        l3 = [r["last3_vs_mean_think"] for r in never if r["last3_vs_mean_think"] is not None]
        phrases = collections.Counter(p for r in R for _, _, p in r["giveup"])
        res[m] = {
            "attempts": n, "accuracy": pct(sum(r["correct"] for r in R), n),
            "stop: submitted / 40-turn cap / 80K budget": f"{len(sub)} / {sum(r['stop'] == 'max_turns' for r in R)} / "
                                                          f"{sum(r['stop'] == 'output_budget' for r in R)}",
            "attempts that never pass a train check": f"{len(never)} ({pct(len(never), n)})",
            "unverified submits (no passing check in the attempt)": f"{len(unver)} ({pct(len(unver), n)}), "
                                                                   f"correct {sum(r['correct'] for r in unver)}",
            "... of them before 40K tokens": f"{len(unver_early)}, correct {sum(r['correct'] for r in unver_early)}",
            "never-pass attempts that run to a cap instead of submitting": sum(not r["submitted"] for r in never),
            "trivial answers (identity / one colour)": f"{sum(r['trivial'] == 'identity' for r in R)} / "
                                                       f"{sum(r['trivial'] == 'one_colour' for r in R)}",
            "attempts with giving-up language": f"{len(gu)} ({pct(len(gu), n)}); before 40K: {len(gu_early)}",
            "giving-up phrases per attempt": round(sum(len(r['giveup']) for r in R) / max(1, n), 2),
            "top phrases": phrases.most_common(8),
            "budget talk in thinking per attempt": round(sum(r["budget_talk"] for r in R) / max(1, n), 2),
            "first train check: median turn / tokens": f"{statistics.median(fct) if fct else '-'} / "
                                                       f"{int(statistics.median(fc)) if fc else '-'}",
            "attempts with no train check at all": sum(r["first_check_turn"] is None for r in R),
            "longest run of turns without a check: median / p90": f"{statistics.median([r['best_streak'] for r in R])} / "
                                                                   f"{sorted(r['best_streak'] for r in R)[int(0.9 * n)]}",
            "share of tokens after the last train check (median, never-pass)": round(statistics.median(
                [r["tail_after_last_check"] for r in never]), 3) if never else None,
            "last-3-turn thinking vs attempt mean (median, never-pass)": round(statistics.median(l3), 2) if l3 else None,
        }
    # stop rule "no passing train check by T tokens -> stop": what it would save and lose (lost = correct attempts
    # whose first pass comes after T, or that never pass; the forced answer is assumed wrong)
    print("\nstop-if-no-pass-by-T: attempts stopped / of them correct (lost) / tokens saved (% of all)")
    for m in MODELS:
        R = by[m]
        tot = sum(r["tokens"] for r in R)
        cells = []
        for T in (40000, 50000, 60000, 70000):
            hit = [r for r in R if r["tokens"] > T and (r["first_pass_tok"] is None or r["first_pass_tok"] > T)]
            cells.append(f"{T // 1000}K: {len(hit)}/{sum(r['correct'] for r in hit)}/{100 * sum(r['tokens'] - T for r in hit) / tot:.1f}%")
        print(f"  {m:9s} " + " | ".join(cells))
    json.dump({"summary": res, "rows": rows}, open(os.path.join(HERE, "results.json"), "w"), default=str)
    keys = list(res[MODELS[0]].keys())
    for k in keys:
        if k == "top phrases":
            continue
        print(f"{k[:62]:62s} " + " | ".join(f"{str(res[m][k]):>18s}" for m in MODELS))
    print("models:", MODELS)
    for m in MODELS:
        print(m, "top phrases:", res[m]["top phrases"])


if __name__ == "__main__":
    main()
