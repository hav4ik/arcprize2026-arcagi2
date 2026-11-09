#!/usr/bin/env python3
"""Step 9: are typos "split-mixing" events? The same identifier is segmented differently by BPE depending on the
character before it (' comps' = comp|s, '(comps' = ( com|ps). Hypothesis: the pruned model emits the first piece(s)
of the segmentation valid in the current context and then continues with a token-aligned suffix of *another*
segmentation of the same word (com + s = 'coms'; '(m' + 'mask' = 'mmask'; ym + min = 'ymmin').

For each typo T of intended I written after context char c:
  A_k = the first k pieces of I's segmentation after c (identifier characters only), k >= 1
  B   = any token-aligned suffix of I's segmentation after any context char in CTX (incl. the whole word)
  T is split-mix-explained if T == A_k + B for some k, B, with T != I.
Chance rate: the same test applied to every OSA-distance-1 variant of I (deletions, insertions of [a-z_0-9],
substitutions, transpositions), averaged per typo. Output: out/splitmix.json and printed table.
"""
import json
import os
import pickle
import string
import sys
from collections import Counter
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from textutil import tk  # noqa: E402

CTX = ["", " ", "(", ".", "=", ",", "[", "\n", "_", "{", ":", "+", "-", "*", "'", "\"", "  ", "\t", ")", "]", "#"]
ALPH = string.ascii_lowercase + string.digits + "_"


@lru_cache(maxsize=None)
def pieces(c, word):
    """Identifier-character pieces of `word` after context char c."""
    e = tk().encode(c + word, add_special_tokens=False)
    out = []
    for (a, b) in e.offsets:
        a2, b2 = max(a, len(c)), b
        if b2 > a2:
            out.append(word[a2 - len(c): b2 - len(c)])
    return tuple(out)


@lru_cache(maxsize=None)
def suffixes(word):
    s = set()
    for c in CTX:
        p = pieces(c, word)
        for m in range(len(p)):
            s.add("".join(p[m:]))
    return frozenset(s)


def explained(t, i, c):
    p = pieces(c, i)
    suf = suffixes(i)
    pre = ""
    for k in range(1, len(p)):
        pre += p[k - 1]
        rest = t[len(pre):]
        if t.startswith(pre) and rest in suf and t != i:
            own = {"".join(p[m:]): m for m in range(len(p))}
            if rest in own:
                return True, ("same-split skip" if own[rest] > k else "same-split repeat")
            return True, "other-split"
    return False, None


def variants(i):
    v = set()
    for k in range(len(i)):
        v.add(i[:k] + i[k + 1:])
        for ch in ALPH:
            v.add(i[:k] + ch + i[k + 1:])
        if k + 1 < len(i):
            v.add(i[:k] + i[k + 1] + i[k] + i[k + 2:])
    for k in range(len(i) + 1):
        for ch in ALPH:
            v.add(i[:k] + ch + i[k:])
    v.discard(i)
    return v


def main():
    lead_char = {"space": " ", "open": "(", "dot": ".", "comma": ",", "op": "=", "linestart": "\n", "other": ""}
    res = {}
    for m in ["E128", "E112", "E96", "E72", "DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95", "API-full", "TRAIN"]:
        f = f"{OUT}/ident_errors_{m}.pkl"
        if not os.path.exists(f):
            continue
        E = [e for e in pickle.load(open(f, "rb")) if e.get("cls") == "typo" and e.get("intended")]
        n = expl = 0
        kinds = Counter()
        chance = []
        ex = Counter()
        for e in E:
            t = e["typo"].split(".")[-1]
            i = e["intended"].split(".")[-1]
            if not i.replace("_", "a").isalnum():
                continue
            c = e.get("lead", "")
            c = c if c in CTX else lead_char.get(e.get("lead_cls"), "")
            ok, kind = explained(t, i, c)
            n += 1
            if ok:
                expl += 1
                kinds[kind] += 1
                ex[(t, i, kind)] += 1
            vs = variants(i)
            chance.append(sum(explained(v, i, c)[0] for v in vs) / max(1, len(vs)))
        res[m] = {"typos": n, "explained": expl, "share": expl / max(1, n),
                  "other-split (needs a different segmentation)": kinds["other-split"],
                  "same-split skip": kinds["same-split skip"], "same-split repeat": kinds["same-split repeat"],
                  "chance share (OSA-1 variants)": sum(chance) / max(1, len(chance)),
                  "top": [f"{a}<-{b} ({k}) x{v}" for (a, b, k), v in ex.most_common(15)]}
        print(m, json.dumps(res[m]), flush=True)
    json.dump(res, open(f"{OUT}/splitmix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
