#!/usr/bin/env python3
"""Step 5b: what is on the line of a SyntaxError? Markers of a typo + in-line self-repair vs other causes."""
import json, os, pickle, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

def err_line(output):
    lines = output.split("\n")
    for j, l in enumerate(lines):
        if l.strip() and set(l.strip()) == {"^"} and j > 0:
            return lines[j - 1]
    return None

def strip_strings(s):
    s = re.sub(r"#[^\n]*", "", s)
    return re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", s)

res = {}
for m in ["API-full", "SH-full", "E128", "E112", "E96", "E72", "TRAIN"]:
    A = pickle.load(open(f"{OUT}/attempts_{m}.pkl", "rb"))
    c = Counter()
    for a in A:
        for t in a["turns"]:
            for x in t["calls"]:
                if x["status"] != "error" or x["exc_type"] != "SyntaxError":
                    continue
                c["n"] += 1
                L = err_line(x["output"]) or ""
                s = strip_strings(L)
                if "does not match" in (x["exc_msg"] or "") and re.search(r"'\{'", x["exc_msg"] or ""):
                    c["{ closed by ] or )"] += 1
                elif "?" in s:
                    c["'?' self-repair marker on the line"] += 1
                elif "if False" in L:
                    c["'if False' self-repair on the line"] += 1
                elif re.search(r"[,(=\[]\s*(\w{1,3}) \1\w+", s):
                    c["word restart ('d dtype', 'z zip')"] += 1
                elif re.match(r"\s*(from|import)\b", L):
                    c["import line (garbled module name)"] += 1
                else:
                    c["other"] += 1
    res[m] = dict(c)
    print(m, dict(c))
json.dump(res, open(f"{OUT}/synerr_lines.json", "w"), indent=1)
