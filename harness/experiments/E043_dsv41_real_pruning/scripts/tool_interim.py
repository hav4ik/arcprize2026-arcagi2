#!/usr/bin/env python3
"""E043: interim/final tool-harness ARC numbers for self-hosted runs vs a reference run on the same attempts.

Per run: attempts done, correct, same-(task, pair, attempt) correct vs the reference, pass@1/pass@2 task-weighted
over tasks whose pairs x attempts are all done, output tokens (mean/median/p90), tool-call error rate, rescues,
stop reasons. Reference default: E030 sub80s40 + minimal header (API full model).

  tool_interim.py e043_e128t_w10_u2_gptq_eng64_sub80 e043_e96t_w10_u2u3_gptq_eng64_sub80 [--ref e030_hdrmin]
"""
import argparse
import collections
import glob
import json
import os
import statistics

RES = os.path.join(os.path.dirname(__file__), "../../../evals/results")


def load(tag):
    fs = glob.glob(os.path.join(RES, f"tools_*_{tag}.jsonl"))
    assert len(fs) == 1, (tag, fs)
    rows = {}
    for line in open(fs[0]):
        r = json.loads(line)
        rows[(r["task_id"], r["pair"], r["attempt"])] = r
    return rows


def task_weighted(rows, attempts):
    by_task = collections.defaultdict(dict)
    for (t, p, a), r in rows.items():
        by_task[t].setdefault(p, {})[a] = bool(r.get("correct"))
    p1, p2 = [], []
    for t, pairs in by_task.items():
        if not all(len(v) == attempts for v in pairs.values()):
            continue
        p1.append(statistics.mean(statistics.mean(v.values()) for v in pairs.values()))
        p2.append(statistics.mean(any(v.values()) for v in pairs.values()))
    return len(p1), (statistics.mean(p1) if p1 else 0), (statistics.mean(p2) if p2 else 0)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def summary(tag, ref, attempts):
    rows = load(tag)
    toks = [r["usage"]["completion"] for r in rows.values()]
    calls = sum(r.get("tool_calls", 0) for r in rows.values())
    errs = sum(r.get("tool_errors", 0) for r in rows.values())
    same = [k for k in rows if k in ref]
    n_tasks, p1, p2 = task_weighted(rows, attempts)
    return {
        "tag": tag,
        "done": len(rows),
        "correct": sum(bool(r.get("correct")) for r in rows.values()),
        "same_attempts": len(same),
        "same_correct": sum(bool(rows[k].get("correct")) for k in same),
        "ref_same_correct": sum(bool(ref[k].get("correct")) for k in same),
        "complete_tasks": n_tasks,
        "pass1_task_weighted": round(100 * p1, 1),
        "pass2_task_weighted": round(100 * p2, 1),
        "out_tokens_mean": round(statistics.mean(toks)) if toks else 0,
        "out_tokens_median": round(statistics.median(toks)) if toks else 0,
        "out_tokens_p90": pct(toks, 0.9),
        "ref_same_out_tokens_mean": round(statistics.mean(ref[k]["usage"]["completion"] for k in same)) if same else 0,
        "turns_mean": round(statistics.mean(r.get("turns", 0) for r in rows.values()), 1) if rows else 0,
        "tool_calls": calls,
        "tool_error_rate": round(100 * errs / max(calls, 1), 1),
        "ref_same_tool_error_rate": round(100 * sum(ref[k].get("tool_errors", 0) for k in same)
                                          / max(sum(ref[k].get("tool_calls", 0) for k in same), 1), 1),
        "rescues": sum(r.get("rescues", 0) or 0 for r in rows.values()),
        "train_pass": sum(bool(r.get("train_pass")) for r in rows.values()),
        "stop_reasons": dict(collections.Counter(r.get("stop_reason") for r in rows.values()).most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default="e030_hdrmin")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = load(a.ref)
    out = [summary(t, ref, a.attempts) for t in [a.ref] + a.tags]
    for s in out:
        print(json.dumps(s))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
