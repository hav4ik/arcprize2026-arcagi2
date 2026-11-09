#!/usr/bin/env python3
"""E043 doubt analysis: rates of doubt / hedge / verification / commit phrases per 1K thinking words, by turn context
(tool harness), from the raw traces. Contexts as in analyze_tool.py (first / explore / after_fail_check /
after_error / post_pass = every turn after the first all-train-pass tool output).

  phrases.py --cache DIR --out results/phrases.json
"""
import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORD_RE, attempt_turns, load_trace  # noqa: E402

PHRASES = {
    # second thoughts
    "wait": r"\b[Ww]ait\b",
    "hmm/hm": r"\b[Hh]m+\b",
    "actually": r"\b[Aa]ctually\b",
    "but_wait/but_let's": r"\b[Bb]ut (?:wait|let'?s|let me)\b",
    # hedges
    "maybe/perhaps": r"\b(?:[Mm]aybe|[Pp]erhaps|[Pp]ossibly)\b",
    "likely/probably": r"\b(?:[Ll]ikely|[Pp]robably)\b",
    "could/might": r"\b(?:[Cc]ould|[Mm]ight)\b",
    "question_mark": r"\?",
    # verification intent in prose
    "need_verify": r"\b[Nn]eed(?: to)? (?:verify|ensure|check|confirm|double-check|make sure|be careful|inspect|test)",
    "let_me_verify": r"\b[Ll]et (?:me|'s|us) (?:also )?(?:verify|check|confirm|double-check|re-?check|re-?examine|"
                     r"reconsider|sanity|make sure|ensure|test)",
    "double/re-check": r"double[- ]?check|\bre-?check|\bre-?verify|verify again|check again",
    "careful/ensure": r"\b(?:careful|ensure|make sure)\b",
    "alternatively": r"\b[Aa]lternatively\b",
    "however": r"\b[Hh]owever\b",
    # commitment
    "commit_words": r"\b(?:[Gg]ood|[Gg]reat|[Pp]erfect|[Ee]xcellent|[Dd]one)\b[.!]",
    "submit": r"\b[Ss]ubmit",
    "all_match": r"\b[Aa]ll (?:training |train |three |two |four |\d )?(?:pairs? |examples? )?(?:match|pass|correct|reproduce)",
}
RX = {k: re.compile(v) for k, v in PHRASES.items()}
CTX = ["first", "explore", "after_fail_check", "after_error", "post_pass"]


def ctx_of(feat, i, fp):
    if i == 0:
        return "first"
    if feat[i - 1]["err"]:
        return "after_error"
    if fp is not None and i > fp:
        return "post_pass"
    if feat[i - 1]["verdict"] == "fail":
        return "after_fail_check"
    return "explore"


def one(a):
    turns, _ = attempt_turns(load_trace(a["trace_path"]))
    feat = a["turns"]
    fp = next((i for i, t in enumerate(feat) if t["verdict"] == "pass"), None)
    cnt = defaultdict(Counter)
    for i, t in enumerate(turns):
        c = ctx_of(feat, i, fp)
        th = t["think"]
        cnt[c]["_words"] += len(WORD_RE.findall(th))
        cnt[c]["_turns"] += 1
        cnt[c]["_tok"] += feat[i]["comp"]
        for k, rx in RX.items():
            cnt[c][k] += len(rx.findall(th))
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["full_api", "full_sh", "E128", "E112", "E96"])
    a = ap.parse_args()
    res = {}
    for m in a.models:
        D = pickle.load(open(os.path.join(a.cache, f"{m}.pkl"), "rb"))
        tot = defaultdict(Counter)
        with Pool(20) as p:
            for cnt in p.imap_unordered(one, D, chunksize=4):
                for c, v in cnt.items():
                    tot[c].update(v)
                    tot["all"].update(v)
        res[m] = {c: {"words": v["_words"], "turns": v["_turns"], "tokens": v["_tok"],
                      "words_per_turn": v["_words"] / max(1, v["_turns"]),
                      **{k: 1000 * v[k] / max(1, v["_words"]) for k in PHRASES}}
                  for c, v in tot.items()}
        res[m]["n_attempts"] = len(D)
    json.dump(res, open(a.out, "w"), indent=1)
    for c in ["all"] + CTX:
        print(f"=== context {c}: per 1K thinking words (words per turn in first row)")
        print("phrase".ljust(20) + "".join(m.rjust(10) for m in a.models))
        print("words/turn".ljust(20) + "".join(f"{res[m].get(c, {}).get('words_per_turn', 0):10.0f}" for m in a.models))
        print("words/attempt".ljust(20) + "".join(f"{res[m].get(c, {}).get('words', 0) / res[m]['n_attempts']:10.0f}"
                                                   for m in a.models))
        for k in PHRASES:
            print(k.ljust(20) + "".join(f"{res[m].get(c, {}).get(k, 0):10.2f}" for m in a.models))


if __name__ == "__main__":
    main()
