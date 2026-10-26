#!/usr/bin/env python3
"""E043 doubt analysis, question 6: frequency of candidate token classes for re-weighting the expert selection.

Every class is defined generically from the rollout text / turn structure (no task ids, no eval data):
  gen            all generated tokens (assistant turns)
  toolblock      tokens outside thinking and content (DSML tool-call blocks; already weighted x10 in tool E128)
  open32_*       first 32 tokens of a turn that follows a tool output, split by what that output was:
                 pass (code compared with train outputs, every pair passes), fail (check failed), err (Python
                 error), other (plain exploration output)
  post1          the whole turn right after the first all-pass tool output ("decision turn")
  post_all       every turn after the first all-pass tool output
  submit_turn    the whole turn that calls submit_answer
  commit_sent    thinking sentences that state success / commit ("All ... match", "Good.", "Perfect", "submit")
  doubt_next16   the 16 tokens that follow a Wait / Hmm / Actually in thinking (where the doubt is resolved)
  err_recovery   the whole turn after a Python error

  train_classes.py --out results/train_classes.json [--runs e031_arc2_train ...] [--limit N]
Also writes the same counts for the dev-split E128 runs and the 33 matching full-model E031 rollouts.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PYERR, TOKENIZER, TRAIN_RUNS, attempt_turns, load_records, load_train_records,  # noqa: E402
                    load_trace, verdict)

COMMIT = re.compile(r"[^.\n]*(?:\b[Aa]ll\b[^.\n]{0,40}\b(?:match|pass|correct|reproduce)|\b(?:Good|Great|Perfect|Excellent)\b"
                    r"|\b[Ss]ubmit)[^.\n]*[.\n]?")
DOUBT = re.compile(r"\b(?:Wait|wait|Hmm+|hmm+|Actually|actually)\b")
_TK = None


def enc(s):
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(TOKENIZER)
    return _TK.encode(s, add_special_tokens=False).ids if s else []


def classes(path):
    try:
        turns, ntr = attempt_turns(load_trace(path))
    except Exception:  # noqa: BLE001
        return None
    c = Counter()
    v = [verdict(t["code"], t["out"], ntr) for t in turns]
    err = [bool(t["out"] and PYERR.search(t["out"])) for t in turns]
    fp = next((i for i, x in enumerate(v) if x == "pass"), None)
    c["rollouts"] = 1
    c["reached_pass"] = int(fp is not None)
    for i, t in enumerate(turns):
        comp = t["comp"]
        th, ct = t["think"], t["content"]
        th_ids = enc(th)
        n_th, n_ct = len(th_ids), len(enc(ct))
        c["gen"] += comp
        c["think"] += n_th
        c["toolblock"] += max(0, comp - n_th - n_ct)
        c["turns"] += 1
        if i > 0:
            kind = "err" if err[i - 1] else ("pass" if v[i - 1] == "pass" else ("fail" if v[i - 1] == "fail" else "other"))
            k = min(32, comp)
            c[f"open32_{kind}"] += k
            c["open32_any"] += k
            c[f"turns_after_{kind}"] += 1
            if err[i - 1]:
                c["err_recovery"] += comp
        if fp is not None and i == fp + 1:
            c["post1"] += comp
            c["post1_submits"] += int("submit_answer" in t["names"])
        if fp is not None and i > fp:
            c["post_all"] += comp
            c["post_turns"] += 1
        if "submit_answer" in t["names"]:
            c["submit_turn"] += comp
        for m in COMMIT.finditer(th):
            c["commit_sent"] += len(enc(m.group(0)))
        for m in DOUBT.finditer(th):
            c["doubt_markers"] += 1
            c["doubt_next16"] += min(16, len(enc(th[m.end():m.end() + 120])))
        # words for marker-rate comparisons
        c["words"] += len(th.split())
    if fp is not None:
        c["post_after_first_submit_turn"] = int(fp + 1 < len(turns) and "submit_answer" in turns[fp + 1]["names"])
    return c


def run(paths, procs):
    tot = Counter()
    with Pool(procs) as p:
        for c in p.imap_unordered(classes, paths, chunksize=8):
            if c:
                tot.update(c)
    return tot


def summarize(tot):
    g = tot["gen"]
    out = {"rollouts": tot["rollouts"], "gen_tokens": g, "gen_per_rollout": g / max(1, tot["rollouts"]),
           "reached_pass": tot["reached_pass"] / max(1, tot["rollouts"]),
           "post1_submit_rate": tot["post1_submits"] / max(1, tot["reached_pass"]),
           "doubt_per_1k_words": 1000 * tot["doubt_markers"] / max(1, tot["words"]), "turns": tot["turns"]}
    for k in ["think", "toolblock", "open32_any", "open32_pass", "open32_fail", "open32_err", "open32_other", "post1",
              "post_all", "submit_turn", "commit_sent", "doubt_next16", "err_recovery"]:
        out[f"{k}_tokens"] = tot[k]
        out[f"{k}_share"] = tot[k] / max(1, g)
    for k in ["turns_after_pass", "turns_after_fail", "turns_after_err", "turns_after_other"]:
        out[k] = tot[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=list(TRAIN_RUNS))
    ap.add_argument("--procs", type=int, default=22)
    a = ap.parse_args()
    res = {}
    allc = Counter()
    for name in a.runs:
        recs = load_train_records(name)
        tot = run([r["trace_path"] for r in recs], a.procs)
        allc.update(tot)
        res[name] = summarize(tot)
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}), flush=True)
    res["all_train"] = summarize(allc)
    # dev check: the 33 dev-split pairs, E128 (sub80s40) vs the full model's own E031 rollouts (s20x60)
    dev_pairs = set()
    for m in ["dev_E128_t1", "dev_E128_minp05", "dev_E128_topp95"]:
        recs = load_records(m)
        dev_pairs |= {(r["task_id"], r["pair"]) for r in recs.values()}
        res[m] = summarize(run([r["trace_path"] for r in recs.values()], a.procs))
    full_dev = [r for r in load_train_records("e031_arc2_train") if (r["task_id"], r["pair"]) in dev_pairs]
    res["full_e031_dev_pairs"] = summarize(run([r["trace_path"] for r in full_dev], a.procs))
    for m in ["full_api", "E128"]:  # eval runs, same classes (diagnostic only)
        recs = load_records(m)
        res[f"eval_{m}"] = summarize(run([r["trace_path"] for r in recs.values()], a.procs))
    json.dump(res, open(a.out, "w"), indent=1)
    keys = list(res["all_train"].keys())
    cols = ["all_train", "full_e031_dev_pairs", "dev_E128_t1", "eval_full_api", "eval_E128"]
    print("metric".ljust(26) + "".join(c[:16].rjust(17) for c in cols))
    for k in keys:
        print(k.ljust(26) + "".join((f"{res[c][k]:17.4f}" if isinstance(res[c][k], float) else str(res[c][k]).rjust(17))
                                    for c in cols))


if __name__ == "__main__":
    main()
