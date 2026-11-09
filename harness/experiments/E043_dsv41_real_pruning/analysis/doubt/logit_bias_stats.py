#!/usr/bin/env python3
"""E043 doubt analysis: token-level statistics for a serving-time logit bias against doubt markers (tool harness).

For candidate token ids (exact ids from the tokenizer), per model: occurrences per 1K thinking tokens, occurrences in
code / submit arguments (a bias applies to the whole generation), and the "doubt episode" length after a marker
(tokens to the next blank line or the end of the turn's thinking).

  logit_bias_stats.py --out results/logit_bias.json [--sample 120]
"""
import argparse
import json
import os
import random
import statistics as st
import sys
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TOKENIZER, attempt_turns, load_records, load_trace  # noqa: E402

CANDIDATES = ["Wait", " Wait", " wait", "Actually", " Actually", " actually", "Hmm", " Hmm", "Maybe", " Maybe", " maybe",
              "Perhaps", " perhaps", "But", " But", "Alternatively", " Alternatively", "Let", " Let"]
_TK = None


def tk():
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(TOKENIZER)
    return _TK


def ids_of():
    out = {}
    for s in CANDIDATES:
        e = tk().encode(s, add_special_tokens=False).ids
        out[s] = e
    return out


def one(path):
    turns, _ = attempt_turns(load_trace(path))
    ids = ids_of()
    single = {s: e[0] for s, e in ids.items() if len(e) == 1}
    th_c, code_c = Counter(), Counter()
    n_th = n_code = 0
    episodes = []
    nl2 = set(tk().encode("\n\n", add_special_tokens=False).ids)
    for t in turns:
        e = tk().encode(t["think"], add_special_tokens=False)
        toks = e.ids
        n_th += len(toks)
        for i, x in enumerate(toks):
            for s, sid in single.items():
                if x == sid:
                    th_c[s] += 1
                    if s.strip() in ("Wait", "wait", "Actually"):
                        j = i + 1
                        while j < len(toks) and toks[j] not in nl2 and "\n\n" not in tk().id_to_token(toks[j]).replace("Ċ", "\n"):
                            j += 1
                        episodes.append(j - i)
        # "Hmm" is two tokens (H + mm): count the pair
        for i in range(len(toks) - 1):
            if toks[i + 1] == ids["Hmm"][1] and toks[i] in (ids["Hmm"][0], ids[" Hmm"][0]):
                th_c["Hmm(2 tok)"] += 1
        c = tk().encode(t["code"] + "\n" + t["submit"], add_special_tokens=False).ids
        n_code += len(c)
        for x in c:
            for s, sid in single.items():
                if x == sid:
                    code_c[s] += 1
        mm = ids["Hmm"][1]
        th_c["mm_token_total"] += sum(1 for x in toks if x == mm)
        code_c["mm_token_total"] += sum(1 for x in c if x == mm)
    return th_c, code_c, n_th, n_code, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=120)
    a = ap.parse_args()
    res = {"ids": ids_of()}
    random.seed(0)
    for m in ["full_api", "E128"]:
        recs = list(load_records(m).values())
        paths = [r["trace_path"] for r in random.sample(recs, min(a.sample, len(recs)))]
        TH, CO, NT, NC, EP = Counter(), Counter(), 0, 0, []
        with Pool(20) as p:
            for th, co, nt, nc, ep in p.imap_unordered(one, paths):
                TH.update(th)
                CO.update(co)
                NT += nt
                NC += nc
                EP += ep
        res[m] = {"attempts": len(paths), "think_tokens": NT, "code_tokens": NC,
                  "per_1k_think_tokens": {k: round(1000 * v / NT, 3) for k, v in TH.items()},
                  "in_code_per_1k_code_tokens": {k: round(1000 * v / max(1, NC), 3) for k, v in CO.items()},
                  "episode_tokens_after_wait_actually": {"n": len(EP), "median": st.median(EP) if EP else None,
                                                          "mean": st.mean(EP) if EP else None,
                                                          "p90": sorted(EP)[int(0.9 * len(EP))] if EP else None}}
        print(m, json.dumps(res[m]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
