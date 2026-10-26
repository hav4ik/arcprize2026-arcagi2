#!/usr/bin/env python3
"""I8 offline estimate: which E050 attempts would get the --post-pass-note, at which turn and token, and what their
post-pass tail (tokens and turns after that tool result) and outcome are.

The trigger is the harness's own detector (evals/run_deepseek_tools.py:train_pass_shown) on each run_python result as
the model saw it, minus the harness notes appended after the output (budget, turns, wall-clock). The note fires on the
first such result, once, when another turn follows (an accepted submit or a cap ends the attempt there). For
comparison, the looser doubt-analysis detector (../doubt/common.py verdict, used for the 14% post-pass bucket in
NOTES.md) is run on the same results. Scores are E050's (analyze.load_arm: the fraction of a task's test pairs right).

  post_pass_reach.py [--runs eff50 eff25]
"""
import argparse
import collections
import gzip
import io
import json
import re
import statistics as st
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "evals"), str(HERE.parent / "doubt"), str(ROOT / "experiments/E050_effort_ab")]
from run_deepseek_tools import train_pass_shown  # noqa: E402
from common import verdict  # noqa: E402
import analyze as e050  # noqa: E402

NOTES = re.compile(r"\n\[(?:budget:|Stop exploring|time:|\d+ turns? left:)")  # appended after the tool output


def raw(out):
    m = NOTES.search(out)
    return out[:m.start()] if m else out


def attempt(trace):
    """Per assistant turn: completion tokens and [(code, raw output)] of its run_python calls."""
    msgs = trace["messages"]
    usage = [e["usage"]["completion_tokens"] for e in trace["events"] if "usage" in e]
    outs = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    n_train = len(re.findall(r"--Example \d+--", next(m["content"] for m in msgs if m["role"] == "user")))
    turns = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        cells = []
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] == "run_python":
                try:
                    code = json.loads(c["function"]["arguments"]).get("code", "")
                except ValueError:
                    code = ""
                cells.append((code, raw(outs.get(c["id"], ""))))
        tok = usage[len(turns)] if len(turns) < len(usage) else 0
        turns.append({"tok": e050.LOOP_MIN_TOKENS if tok >= e050.BIG else tok, "cells": cells})
    return turns, n_train


def first(turns, hit):
    """1-based turn of the first run_python result that `hit`s, if another turn follows it; else None."""
    for k, t in enumerate(turns, 1):
        if any(hit(code, out) for code, out in t["cells"]):
            return k if k < len(turns) else None
    return None


def summarize(name, rows):
    tot = sum(r["tokens"] for r in rows)
    fired = [r for r in rows if r["k"] is not None]
    loose = [r for r in rows if r["k_loose"] is not None]
    post = sum(r["post"] for r in fired)
    print(f"\n== {name}: {len(rows)} attempts (one rollout per task), {tot / 1e3:.0f}K output tokens, "
          f"mean score {st.mean(r['score'] for r in rows):.3f}")
    print(f"note fires in {len(fired)} attempts; looser doubt detector: {len(loose)} "
          f"(post-pass {sum(r['post_loose'] for r in loose) / tot:.1%} of output)")
    if not fired:
        return
    q = lambda xs: "/".join(f"{x:.0f}" for x in st.quantiles(xs, n=4)) if len(xs) > 1 else str(xs)
    print(f"  fires at turn (q1/med/q3) {q([r['k'] for r in fired])}, at {q([r['at'] / 1e3 for r in fired])}K tokens; "
          f"{sum(r['k'] == r['k_loose'] for r in fired)} on the same turn as the looser detector, "
          f"{sum(r['k_loose'] is not None and r['k'] > r['k_loose'] for r in fired)} later")
    print(f"  after the note: {post / 1e3:.0f}K tokens = {post / tot:.1%} of all output, "
          f"{q([r['post'] / 1e3 for r in fired])}K per attempt, {st.mean(r['after'] for r in fired):.1f} turns")
    print(f"  submitted after the note {sum(r['submitted'] for r in fired)}/{len(fired)}; "
          f"mean score {st.mean(r['score'] for r in fired):.3f} (not fired: "
          f"{st.mean([r['score'] for r in rows if r['k'] is None] or [0]):.3f})")
    for lo, hi in ((-1, 2000), (2000, 10000), (10000, 10 ** 9)):  # tails of <= 2K, 2-10K, > 10K tokens
        b = [r for r in fired if lo < r["post"] <= hi]
        if b:
            print(f"  tail {max(lo, 0) // 1000}-{hi // 1000 if hi < 10 ** 9 else 'inf'}K: {len(b)} attempts, "
                  f"{sum(r['post'] for r in b) / max(post, 1):.0%} of post-note tokens, "
                  f"mean score {st.mean(r['score'] for r in b):.2f}, "
                  f"later all-pass results {st.mean(r['later_hits'] for r in b):.1f}")
    missed = [r for r in loose if r["k"] is None]
    print(f"  looser-detector passes without a note: {len(missed)} attempts, "
          f"{sum(r['post_loose'] for r in missed) / 1e3:.0f}K post-pass tokens, "
          f"mean score {st.mean([r['score'] for r in missed] or [0]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["eff50"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    ch, sol = e050.kaggle_set()
    out = {}
    for run in a.runs:
        d = ROOT / "experiments/E050_effort_ab/runs" / run
        arm = e050.load_arm(d / "dsv41_logs/attempts.jsonl", ch, sol)
        rows = []
        with tarfile.open(d / "dsv41_traces.tar.gz") as tf:
            for m in sorted(tf.getmembers(), key=lambda m: m.name):
                if not m.name.endswith("_p0_a0.json.gz"):
                    continue
                task = Path(m.name).name.split("_")[0]
                trace = json.load(gzip.open(io.BytesIO(tf.extractfile(m).read())))
                turns, n_train = attempt(trace)
                k = first(turns, lambda c, o: train_pass_shown(c, o, n_train))
                kl = first(turns, lambda c, o: verdict(c, o, n_train) == "pass")
                r = arm[task]
                rows.append({"task": task, "k": k, "k_loose": kl, "score": r["score"], "tokens": r["tokens"],
                             "turns": len(turns), "submitted": r["source"] == "submit",
                             "at": sum(t["tok"] for t in turns[:k]) if k else None,
                             "post": sum(t["tok"] for t in turns[k:]) if k else 0,
                             "post_loose": sum(t["tok"] for t in turns[kl:]) if kl else 0,
                             "after": len(turns) - k if k else 0,
                             "later_hits": sum(any(train_pass_shown(c, o, n_train) for c, o in t["cells"])
                                               for t in turns[k:]) if k else 0})
        summarize(run, rows)
        out[run] = rows
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
