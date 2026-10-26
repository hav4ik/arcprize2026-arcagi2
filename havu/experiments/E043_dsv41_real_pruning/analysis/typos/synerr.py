#!/usr/bin/env python3
"""Step 5: SyntaxError taxonomy, bracket errors at merged bracket tokens, and code-corruption markers per model.

Per model prints/saves (out/synerr.json):
  - SyntaxError message classes per 100 run_python calls;
  - bracket-mismatch opener->closer pairs;
  - the BPE token under the caret: share that is a merged multi-character bracket token (e.g. '])', '))', '())');
    baseline = share of closing-bracket characters in all code that sit in merged bracket tokens;
  - markers per 100 calls: 'if False else' self-repair, '?' outside strings/comments, CJK characters in code.
"""
import json
import os
import pickle
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from textutil import encode  # noqa: E402

MODELS = ["API-full", "SH-full", "E128", "E112", "E96", "E72", "DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95",
          "TRAIN"]
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
BR = set("()[]{}")


def msg_class(m):
    m = m or ""
    if "does not match opening parenthesis" in m:
        return "bracket-mismatch"
    if m.startswith("unmatched"):
        return "unmatched-closer"
    if "was never closed" in m:
        return "never-closed"
    if "forgot a comma" in m:
        return "forgot-comma"
    if "string literal" in m or "f-string" in m or "EOF while scanning" in m:
        return "string"
    if "incomplete input" in m:
        return "incomplete-input"
    if m.startswith("invalid syntax"):
        return "invalid-syntax"
    if "line continuation" in m:
        return "line-continuation"
    if "assign" in m or "expression" in m:
        return "bad-assign/expr"
    return "other"


def strip_strings(code):
    code = re.sub(r"#[^\n]*", "", code)
    return re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", code)


def caret_token(output):
    lines = output.split("\n")
    for j, l in enumerate(lines):
        if l.strip() and set(l.strip()) == {"^"} and j > 0:
            line = lines[j - 1]
            col = len(l) - len(l.lstrip())
            ids, offs, toks = encode(line)
            for t, (a, b) in zip(toks, offs):
                if a <= col < b:
                    return t, line[col:col + 1]
            return None, None
    return None, None


def main():
    res = {}
    for M in MODELS:
        A = pickle.load(open(f"{OUT}/attempts_{M}.pkl", "rb"))
        calls = [c for a in A for t in a["turns"] for c in t["calls"] if c["name"] == "run_python"]
        n = len(calls)
        cls = Counter()
        pairs = Counter()
        caret = Counter()
        caret_merged = caret_n = 0
        markers = Counter()
        br_chars = br_merged = 0
        for i, c in enumerate(calls):
            code = c["code"] or ""
            if "if False else" in code:
                markers["if-False-else"] += 1
            if "?" in strip_strings(code):
                markers["?-in-code"] += 1
            if CJK.search(code):
                markers["CJK-in-code"] += 1
            if CJK.search(strip_strings(code)):
                markers["CJK-outside-strings"] += 1
            if i % 7 == 0:  # baseline sample for merged closing-bracket tokens
                ids, offs, toks = encode(code)
                for t, (a, b) in zip(toks, offs):
                    nb = sum(1 for ch in code[a:b] if ch in ")]}")
                    if nb:
                        br_chars += nb
                        if sum(1 for ch in code[a:b] if ch in BR) >= 2:
                            br_merged += nb
            if c["status"] == "error" and c["exc_type"] in ("SyntaxError", "IndentationError", "TabError"):
                k = msg_class(c["exc_msg"]) if c["exc_type"] == "SyntaxError" else "indentation"
                cls[k] += 1
                m = re.match(r"closing parenthesis '(.)' does not match opening parenthesis '(.)'", c["exc_msg"] or "")
                if m:
                    pairs[f"{m.group(2)}..{m.group(1)}"] += 1
                if k in ("bracket-mismatch", "unmatched-closer"):
                    t, ch = caret_token(c["output"])
                    if t is not None:
                        caret_n += 1
                        caret[t] += 1
                        if sum(1 for x in t if x in BR) >= 2:
                            caret_merged += 1
        res[M] = {"run_python_calls": n,
                  "syntax_per100": {k: round(100 * v / n, 2) for k, v in cls.most_common()},
                  "syntax_counts": dict(cls.most_common()),
                  "bracket_pairs": dict(pairs.most_common()),
                  "caret_tokens_top": caret.most_common(12),
                  "caret_merged_share": round(caret_merged / caret_n, 3) if caret_n else None, "caret_n": caret_n,
                  "baseline_closer_in_merged_token": round(br_merged / max(1, br_chars), 3),
                  "markers_per100": {k: round(100 * v / n, 2) for k, v in markers.items()},
                  "markers": dict(markers)}
        print(M, json.dumps(res[M]), flush=True)
    json.dump(res, open(f"{OUT}/synerr.json", "w"), indent=1)


if __name__ == "__main__":
    main()
