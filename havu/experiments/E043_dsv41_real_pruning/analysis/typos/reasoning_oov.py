#!/usr/bin/env python3
"""Step 8: are misspellings confined to code? Near-miss out-of-vocabulary words in *reasoning* text.

Vocabulary V = every word ([A-Za-z_][A-Za-z0-9_]{3,}) in the full-model training rollouts (TRAIN: reasoning + code +
tool outputs + user prompts are not stored, so reasoning/code/outputs). A reasoning word of an attempt is a
near-miss OOV if it is not in V, not anywhere in that attempt's code or tool outputs, and within OSA distance 1 of a
word of V that has length >= 5 (a misspelling of a known word, not a new name); words with digits and case-only
differences are excluded.
Rates per 10k reasoning words, per model; examples with their canonical tokens.
"""
import json
import os
import pickle
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from textutil import encode, osa  # noqa: E402

W = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
MODELS = ["API-full", "SH-full", "E128", "E112", "E96", "E72", "DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95"]


def main():
    T = pickle.load(open(f"{OUT}/attempts_TRAIN.pkl", "rb"))
    V = Counter()
    for a in T:
        for t in a["turns"]:
            V.update(W.findall(t["reasoning"]))
            for c in t["calls"]:
                V.update(W.findall(c["code"] or ""))
                V.update(W.findall(c["output"] or ""))
    vocab = set(V)
    index = {}
    for w in vocab:
        if len(w) >= 5 and V[w] >= 3:
            index.setdefault(w, set()).add(w)
            for i in range(len(w)):
                index.setdefault(w[:i] + w[i + 1:], set()).add(w)
    near_cache = {}

    def near(w):
        if w in near_cache:
            return near_cache[w]
        cands = set(index.get(w, ()))
        for i in range(len(w)):
            cands |= index.get(w[:i] + w[i + 1:], set())
        best = None
        if any(ch.isdigit() for ch in w):
            near_cache[w] = None
            return None
        for v in sorted(cands, key=lambda x: -V[x]):
            if v != w and v.lower() != w.lower() and not any(ch.isdigit() for ch in v) and osa(w, v) == 1:
                best = v
                break
        near_cache[w] = best
        return best

    res = {}
    rng = random.Random(0)
    for m in MODELS:
        A = pickle.load(open(f"{OUT}/attempts_{m}.pkl", "rb"))
        n_words = 0
        oov = Counter()
        nm = Counter()
        code_words = 0
        code_nm = Counter()
        for a in A:
            local = set()
            for t in a["turns"]:
                for c in t["calls"]:
                    local.update(W.findall(c["code"] or ""))
                    local.update(W.findall(c["output"] or ""))
            for t in a["turns"]:
                ws = W.findall(t["reasoning"])
                n_words += len(ws)
                for w in ws:
                    if w in vocab or w in local:
                        continue
                    oov[w] += 1
                    v = near(w)
                    if v:
                        nm[(w, v)] += 1
        from textutil import token_event
        ev = Counter()
        for (w, v), k in nm.items():
            e, j = token_event(w, v, encode(" " + v)[2])
            ev[e] += k
        ex = rng.sample(sorted(nm), min(25, len(nm)))
        res[m] = {"reasoning_words": n_words, "oov_per10k": 1e4 * sum(oov.values()) / max(1, n_words),
                  "near_miss_per10k": 1e4 * sum(nm.values()) / max(1, n_words),
                  "near_miss_token_events": {k: round(v / max(1, sum(ev.values())), 3) for k, v in ev.most_common()},
                  "near_miss_n": sum(nm.values()),
                  "examples": [(w, v, encode(" " + v)[2], encode(" " + w)[2]) for w, v in ex]}
        print(m, json.dumps({k: v for k, v in res[m].items() if k != "examples"}), flush=True)
        for e in res[m]["examples"][:12]:
            print("    ", e)
    json.dump(res, open(f"{OUT}/reasoning_oov.json", "w"), indent=1)


if __name__ == "__main__":
    main()
