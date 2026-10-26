#!/usr/bin/env python3
"""Step 3: count candidate token classes (as *targets*, i.e. the token being predicted) in generated text.

Generic definitions (text/token only, no parsing):
  word        = maximal run [A-Za-z_][A-Za-z0-9_]* (identifiers and English words; pure numbers excluded)
  WIC         = word-internal continuation: the token starts strictly inside a word (2nd+ token of a multi-token word)
  WIC-alnum   = WIC whose boundary splits two alphanumerics (e.g. com|ps, sc|ipy, de|que); excludes _-boundaries (_info)
  copied      = the word occurred earlier in the context (system/tool description, user prompt, earlier generated
                text, tool outputs, or earlier in the same segment)
  retok-copy  = copied, but this occurrence's token ids differ from every earlier occurrence's ids (re-spelling)
  def-site    = tokens of words on an import line, or of the name after def/class, or of an assignment target
Segments: 'reason' (reasoning_content + content) and 'code' (tool-call code: the x10 tool-block region).
Approximation: segments are tokenized separately (the real rendering adds DSML tags around code).

Writes out/tokclass_<model>.json with counts per (segment, class).
"""
import json
import os
import pickle
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from textutil import encode  # noqa: E402

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TOOL_DESC = ("Execute a Python cell in your persistent IPython session. Variables/functions persist. Preloaded: train "
             "(list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. scipy is available. "
             "submit_answer code solve test_input answer")
DEF_LINE = re.compile(r"^\s*(?:import\s|from\s|def\s|class\s)")
ASSIGN = re.compile(r"^\s*([A-Za-z_][\w,\s]*?)\s*(?<![=!<>])=(?!=)", re.M)


def classify_segment(text, seg, seen, counts, extra):
    ids, offs, _ = encode(text)
    if not ids:
        return
    words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text)]
    # def-site char mask
    defmask = None
    if seg == "code":
        defmask = bytearray(len(text))
        pos = 0
        for line in text.splitlines(keepends=True):
            if DEF_LINE.match(line):
                defmask[pos:pos + len(line)] = b"\x01" * len(line)
            else:
                m = ASSIGN.match(line)
                if m:
                    defmask[pos + m.start(1):pos + m.end(1)] = b"\x01" * (m.end(1) - m.start(1))
            pos += len(line)
    wi = 0
    occ_tok = {}  # word index -> list of token idx
    tok_word = [-1] * len(ids)
    for k, (a, b) in enumerate(offs):
        while wi < len(words) and words[wi][1] <= a:
            wi += 1
        if wi < len(words) and words[wi][0] < b and a < words[wi][1]:
            tok_word[k] = wi
            occ_tok.setdefault(wi, []).append(k)
    word_info = {}
    for w, ks in occ_tok.items():
        s, e, name = words[w]
        tid = tuple(ids[k] for k in ks)
        prev = seen.get(name)
        copied = prev is not None
        retok = copied and tid not in prev
        word_info[w] = (copied, retok, len(ks))
        seen.setdefault(name, set()).add(tid)
    for k, (a, b) in enumerate(offs):
        counts[(seg, "all")] += 1
        w = tok_word[k]
        if w < 0:
            continue
        s, e, name = words[w]
        copied, retok, ntok = word_info[w]
        counts[(seg, "word")] += 1
        if a > s:  # starts strictly inside the word
            counts[(seg, "WIC")] += 1
            alnum = text[a - 1].isalnum() and text[a].isalnum()
            if alnum:
                counts[(seg, "WIC-alnum")] += 1
            if copied:
                counts[(seg, "WIC-copied")] += 1
                if alnum:
                    counts[(seg, "WIC-alnum-copied")] += 1
                if retok:
                    counts[(seg, "WIC-retok-copy")] += 1
                    if alnum:
                        counts[(seg, "WIC-alnum-retok-copy")] += 1
            else:
                counts[(seg, "WIC-fresh")] += 1
        else:
            counts[(seg, "word-first")] += 1
        if ntok >= 2:
            counts[(seg, "multitok-word-token")] += 1
        if copied:
            counts[(seg, "copied-word-token")] += 1
            if retok:
                counts[(seg, "retok-copy-token")] += 1
        if defmask is not None and s < len(defmask) and defmask[s]:
            counts[(seg, "def-site")] += 1
            if a > s:
                counts[(seg, "def-site-WIC")] += 1
    extra["tokens"] += len(ids)


def register(text, seen):
    if not text:
        return
    ids, offs, _ = encode(text[:20000])
    words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text[:20000])]
    wi = 0
    cur = {}
    for k, (a, b) in enumerate(offs):
        while wi < len(words) and words[wi][1] <= a:
            wi += 1
        if wi < len(words) and words[wi][0] < b and a < words[wi][1]:
            cur.setdefault(wi, []).append(ids[k])
    for w, t in cur.items():
        seen.setdefault(words[w][2], set()).add(tuple(t))


def count_attempt(a):
    counts = Counter()
    extra = Counter()
    seen = {}
    for m in WORD.finditer(TOOL_DESC):
        seen.setdefault(m.group(), set()).add(tuple(encode(" " + m.group())[0]))
    for t in a["turns"]:
        classify_segment(t["reasoning"] + "\n" + t["content"], "reason", seen, counts, extra)
        for c in t["calls"]:
            classify_segment(c["code"] or "", "code", seen, counts, extra)
            # tool output joins the context (not generated): register its words with their token ids
            register(c["output"] or "", seen)
    return counts


def run(model):
    A = pickle.load(open(f"{OUT}/attempts_{model}.pkl", "rb"))
    tot = Counter()
    with ProcessPoolExecutor(20) as ex:
        for c in ex.map(count_attempt, A, chunksize=8):
            tot.update(c)
    out = {f"{k[0]}|{k[1]}": v for k, v in sorted(tot.items())}
    json.dump(out, open(f"{OUT}/tokclass_{model}.json", "w"), indent=1)
    print(model, json.dumps(out), flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m)
