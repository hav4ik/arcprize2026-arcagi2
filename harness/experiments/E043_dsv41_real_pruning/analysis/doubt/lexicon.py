#!/usr/bin/env python3
"""E043 doubt analysis: data-driven lexical signature of pruned-model thinking vs the full model (tool harness).

  lexicon.py --out results/lexicon.json [--models full_api E128 E96]

Per model, over all thinking text: unigram and bigram rates per 1K words (lower-cased), sentence-initial words per
1K sentences, and the first two words of each turn. Prints the most over- and under-used items vs full_api
(log2 ratio, min count 150 in the pooled pair).
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import attempt_turns, load_records, load_trace  # noqa: E402

TOK = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[?!]")
SENT = re.compile(r"(?:^|(?<=[.!?:])\s+|\n+)([A-Z][A-Za-z']*)")


def one(path):
    turns, _ = attempt_turns(load_trace(path))
    uni, bi, sent, open_ = Counter(), Counter(), Counter(), Counter()
    words = 0
    for t in turns:
        th = t["think"]
        w = [x.lower() for x in TOK.findall(th)]
        words += len(w)
        uni.update(w)
        bi.update(zip(w, w[1:]))
        sent.update(m.group(1) for m in SENT.finditer(th))
        open_[" ".join(th.split()[:2])] += 1
    return uni, bi, sent, open_, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["full_api", "E128", "E112", "E96"])
    a = ap.parse_args()
    agg = {}
    for m in a.models:
        paths = [r["trace_path"] for r in load_records(m).values()]
        U, B, S, O = Counter(), Counter(), Counter(), Counter()
        W = 0
        with Pool(20) as p:
            for u, b, s, o, w in p.imap_unordered(one, paths, chunksize=4):
                U.update(u)
                B.update(b)
                S.update(s)
                O.update(o)
                W += w
        agg[m] = {"U": U, "B": B, "S": S, "O": O, "W": W, "NS": sum(S.values()), "NO": sum(O.values())}
    ref = agg[a.models[0]]
    res = {}
    for m in a.models[1:]:
        x = agg[m]
        out = {}
        for name, key, den in [("unigram_per_1k_words", "U", "W"), ("bigram_per_1k_words", "B", "W"),
                               ("sentence_start_per_1k_sentences", "S", "NS"), ("turn_opening_per_1k_turns", "O", "NO")]:
            rows = []
            for k in set(ref[key]) | set(x[key]):
                c1, c2 = ref[key][k], x[key][k]
                if c1 + c2 < (150 if key != "O" else 40):
                    continue
                r1, r2 = 1000 * c1 / ref[den], 1000 * c2 / x[den]
                lr = math.log2((r2 + 1e-3) / (r1 + 1e-3))
                rows.append((" ".join(k) if isinstance(k, tuple) else k, round(r1, 3), round(r2, 3), round(lr, 2), c1, c2))
            rows.sort(key=lambda r: -r[3])
            out[name] = {"over": rows[:40], "under": rows[-40:][::-1]}
        res[m] = out
        print(f"===== {m} vs {a.models[0]}  (words {x['W']} vs {ref['W']})")
        for name in out:
            print(f"--- {name}: over-used")
            for r in out[name]["over"][:25]:
                print(f"   {r[0]:28s} {r[1]:8.3f} -> {r[2]:8.3f}  log2 {r[3]:+.2f}")
            print(f"--- {name}: under-used")
            for r in out[name]["under"][:25]:
                print(f"   {r[0]:28s} {r[1]:8.3f} -> {r[2]:8.3f}  log2 {r[3]:+.2f}")
    # top sentence starts and turn openings, per model
    res["top"] = {m: {"sentence_start": [(k, round(1000 * v / agg[m]["NS"], 2)) for k, v in agg[m]["S"].most_common(40)],
                      "turn_opening": [(k, round(1000 * v / agg[m]["NO"], 2)) for k, v in agg[m]["O"].most_common(30)]}
                  for m in a.models}
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
