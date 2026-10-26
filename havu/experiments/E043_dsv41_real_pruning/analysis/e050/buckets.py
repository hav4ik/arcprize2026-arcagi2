#!/usr/bin/env python3
"""E050 (effort 25 vs 50 on E128, 1 attempt per task): where do the two arms spend their tokens?

Same split as ../doubt/analyze_tool.py, on the E050 traces (one rollout per task, multi-test tasks included):
  (a) error recovery    turns right after a Python error
  (b) post-pass         later turns after the first all-train-pass tool output, minus (a)
  (c) pre-pass          turns up to and including the first pass, minus (a), in attempts that pass
  (c') never-pass       all non-(a) turns of attempts that never pass
Plus turn-1 size, thinking share, doubt markers, giving-up language, stops and the tokens/turns to the first pass.
Correctness is E050's (analyze.load_arm). Paired by task; 95% bootstrap CIs over tasks.

  buckets.py --traces <dir with eff25/ and eff50/ extracted from dsv41_traces.tar.gz> --out results.json
"""
import argparse
import collections
import gzip
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/vu/Workspace/ARC-AGI-2"
sys.path.insert(0, os.path.join(HERE, "..", "doubt"))
sys.path.insert(0, os.path.join(HERE, "..", "giving_up"))
sys.path.insert(0, f"{ROOT}/experiments/E050_effort_ab")
from common import PYERR, TOKENIZER, WORD_RE, attempt_turns, core_spans, is_train_check, verdict  # noqa: E402
from giving_up import GIVEUP  # noqa: E402
import analyze as e050  # noqa: E402

ARMS = ("eff25", "eff50")
_TK = None


def ntok(s):
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(TOKENIZER)
    return len(_TK.encode(s, add_special_tokens=False).ids) if s else 0


def summarize(path):
    turns, ntr = attempt_turns(json.load(gzip.open(path)))
    T = []
    for t in turns:
        T.append({"comp": t["comp"] or 0, "think": ntok(t["think"]), "err": bool(t["out"] and PYERR.search(t["out"])),
                  "check": bool(t["code"] and is_train_check(t["code"])), "v": verdict(t["code"], t["out"], ntr),
                  "words": len(WORD_RE.findall(t["think"])), "core": len(core_spans(t["think"])),
                  "giveup": len(GIVEUP.findall(t["think"] + "\n" + t["content"]))})
    fp = next((i for i, t in enumerate(T) if t["v"] == "pass"), None)
    rec = lambda i: i > 0 and T[i - 1]["err"]
    s = {"comp": sum(t["comp"] for t in T), "turns": len(T), "fp": fp,
         "a_rec": sum(T[i]["comp"] for i in range(len(T)) if rec(i)),
         "b_post": sum(T[i]["comp"] for i in range(len(T)) if fp is not None and i > fp and not rec(i)),
         "c_pre": sum(T[i]["comp"] for i in range(len(T)) if fp is not None and i <= fp and not rec(i)),
         "c_never": sum(T[i]["comp"] for i in range(len(T)) if fp is None and not rec(i)),
         "turn1": T[0]["comp"] if T else 0, "later_turn_mean": st.mean(t["comp"] for t in T[1:]) if len(T) > 1 else 0,
         "think": sum(t["think"] for t in T), "errs": sum(t["err"] for t in T), "checks": sum(t["check"] for t in T),
         "fail_checks": sum(t["v"] == "fail" for t in T), "words": sum(t["words"] for t in T),
         "core": sum(t["core"] for t in T), "giveup": sum(t["giveup"] for t in T),
         "tok_to_pass": sum(t["comp"] for t in T[:fp + 1]) if fp is not None else None,
         "turns_to_pass": fp + 1 if fp is not None else None,
         "turns_post": len(T) - fp - 1 if fp is not None else None,
         "rec_turns": sum(rec(i) for i in range(len(T)))}
    return s


def boot(xs, reps=4000, seed=0):
    rng = random.Random(seed)
    m = [st.mean(rng.choice(xs) for _ in xs) for _ in range(reps)]
    m.sort()
    return st.mean(xs), m[int(0.025 * reps)], m[int(0.975 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()
    ch, sol = e050.kaggle_set()
    rows = {arm: e050.load_arm(f"{ROOT}/experiments/E050_effort_ab/runs/{arm}/dsv41_logs/attempts.jsonl", ch, sol)
            for arm in ARMS}
    S = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for t, r in rows[arm].items():
            p = os.path.join(a.traces, arm, "traces", f"{t}_p0_a0.json.gz")
            if os.path.exists(p):
                s = summarize(p)
                s.update(score=r["score"], stop=r["stop"], train_pass=r["train_pass"], usage=r["tokens"], n=r["n"])
                S[arm][t] = s
    tasks = sorted(set(S["eff25"]) & set(S["eff50"]))
    out = {"tasks": len(tasks)}
    print(f"{len(tasks)} tasks with traces in both arms")

    # sanity: per-turn sums vs logged usage; detector vs harness train_pass
    for arm in ARMS:
        R = [S[arm][t] for t in tasks]
        resid = sum(abs(s["usage"] - s["comp"]) for s in R)
        agree = collections.Counter((s["fp"] is not None, bool(s["train_pass"])) for s in R)
        out[f"sanity_{arm}"] = {"abs_resid_tokens": resid, "detector_vs_train_pass": {str(k): v for k, v in agree.items()}}
        print(f"  {arm}: |usage - sum of turns| = {resid}; (detector pass, harness train_pass): {dict(agree)}")

    def table(label, keys):
        print(f"\n{label}: mean per task, effort 25 / effort 50 / diff [95% CI]")
        res = {}
        for k in keys:
            d = [S["eff25"][t][k] - S["eff50"][t][k] for t in tasks]
            m25, m50 = st.mean(S["eff25"][t][k] for t in tasks), st.mean(S["eff50"][t][k] for t in tasks)
            md, lo, hi = boot(d)
            res[k] = {"eff25": m25, "eff50": m50, "diff": md, "ci": [lo, hi]}
            print(f"  {k:16s} {m25:9.1f} {m50:9.1f}   {md:+8.1f} [{lo:+.1f}, {hi:+.1f}]")
        return res

    out["buckets"] = table("token buckets", ["comp", "a_rec", "b_post", "c_pre", "c_never"])
    out["other"] = table("other per-task means", ["turns", "turn1", "later_turn_mean", "think", "errs", "rec_turns",
                                                  "checks", "fail_checks", "core", "giveup", "score"])
    for arm in ARMS:
        R = [S[arm][t] for t in tasks]
        reached = [s for s in R if s["fp"] is not None]
        out[arm] = {
            "think_share": sum(s["think"] for s in R) / sum(s["comp"] for s in R),
            "core_per_1k_words": 1000 * sum(s["core"] for s in R) / max(1, sum(s["words"] for s in R)),
            "reach_pass": len(reached), "never_pass": len(R) - len(reached),
            "never_pass_tokens_mean": st.mean(s["comp"] for s in R if s["fp"] is None),
            "tok_to_pass_median": st.median(s["tok_to_pass"] for s in reached),
            "turns_to_pass_median": st.median(s["turns_to_pass"] for s in reached),
            "turns_post_mean": st.mean(s["turns_post"] for s in reached),
            "stops": dict(collections.Counter(s["stop"] for s in R)),
            "never_pass_capped": sum(s["fp"] is None and s["stop"] != "submitted" for s in R),
            "score_reached": st.mean(s["score"] for s in reached),
            "score_never": st.mean(s["score"] for s in R if s["fp"] is None),
        }
    print("\nper arm:")
    for k in out["eff25"]:
        print(f"  {k:24s} {str(round(out['eff25'][k], 3) if isinstance(out['eff25'][k], float) else out['eff25'][k]):>34s}"
              f" {str(round(out['eff50'][k], 3) if isinstance(out['eff50'][k], float) else out['eff50'][k]):>34s}")
    # paired on tasks where both arms reach a pass: does 25 get there sooner or stop sooner after?
    both = [t for t in tasks if S["eff25"][t]["fp"] is not None and S["eff50"][t]["fp"] is not None]
    print(f"\n{len(both)} tasks reach a pass in both arms (25 - 50 [95% CI]):")
    out["both_pass"] = {"tasks": len(both)}
    for k in ("tok_to_pass", "turns_to_pass", "b_post", "turns_post", "comp"):
        md, lo, hi = boot([S["eff25"][t][k] - S["eff50"][t][k] for t in both])
        out["both_pass"][k] = {"diff": md, "ci": [lo, hi]}
        print(f"  {k:14s} {md:+8.1f} [{lo:+.1f}, {hi:+.1f}]")
    trans = collections.Counter((S["eff25"][t]["fp"] is not None, S["eff50"][t]["fp"] is not None) for t in tasks)
    out["reach_transitions"] = {f"25:{k[0]} 50:{k[1]}": v for k, v in trans.items()}
    print("reach-a-pass (eff25, eff50):", dict(trans))
    json.dump({"summary": out, "per_task": S}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
