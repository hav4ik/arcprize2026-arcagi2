#!/usr/bin/env python3
"""E043 doubt analysis, question 5: the no-tool harness (e043nt_* runs), reported separately from tool statistics.

  analyze_notool.py --out results/notool_stats.json

Paired on the same (task, pair, attempt) keys as the full model: completion and reasoning tokens, words, the same
phrase classes as phrases.py per 1K reasoning words, 6-gram repetition, and the share of reasoning written after the
final answer grid first appears in full (rows matched in order on the digit stream; correct attempts only).
"""
import argparse
import gzip
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NOTOOL_RUNS, RES, ROOT, WORD_RE  # noqa: E402
from phrases import PHRASES  # noqa: E402

RX = {k: re.compile(v) for k, v in PHRASES.items()}
LETME = re.compile(r"\bLet me\b")
NEEDS = re.compile(r"(?:^|[.\n]\s*)Need\b")


def load(tag):
    out = {}
    for line in open(f"{RES}/{tag}.jsonl"):
        r = json.loads(line)
        if "error" in r or "task_id" not in r:
            continue
        p = r.get("reasoning_path")
        fp = p if p and os.path.isabs(p) else os.path.join(ROOT, p or "")
        if not p or not os.path.exists(fp):
            continue
        r["_text"] = gzip.open(fp, "rt").read()
        out[(r["task_id"], r["pair"], r["attempt"])] = r
    return out


def answer_pos(text, pred):
    """Character index where the final answer grid has been written in full for the first time (rows in order)."""
    if not (isinstance(pred, list) and pred and all(isinstance(r, list) for r in pred)):
        return None
    idx = [i for i, ch in enumerate(text) if ch.isdigit()]
    ds = "".join(text[i] for i in idx)
    rows = ["".join(str(c) for c in row) for row in pred]
    if sum(len(r) for r in rows) < 12:
        return None
    # earliest completion: try each start of row 0, match the remaining rows in order within a window
    best = None
    start = 0
    while True:
        j = ds.find(rows[0], start)
        if j < 0:
            break
        k = j + len(rows[0])
        ok = True
        for r in rows[1:]:
            n = ds.find(r, k, k + len(r) + 40)  # rows must follow closely (row labels may add a few digits)
            if n < 0:
                ok = False
                break
            k = n + len(r)
        if ok:
            best = idx[k - 1]
            break
        start = j + 1
    return best


def feats(r):
    t = r["_text"]
    u = r.get("usage") or {}
    w = WORD_RE.findall(t)
    words = [x.lower() for x in w]
    grams = [tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
    pos = answer_pos(t, r.get("pred")) if r.get("correct") else None
    return {"tok": u.get("completion_tokens") or 0,
            "rtok": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "words": len(w), "correct": bool(r.get("correct")), "finish": r.get("finish_reason"),
            "ph": {k: len(rx.findall(t)) for k, rx in RX.items()}, "letme": len(LETME.findall(t)),
            "need": len(NEEDS.findall(t)),
            "rep6": (1 - len(set(grams)) / len(grams)) if len(grams) > 50 else None,
            "after_answer_share": (len(t) - pos) / len(t) if pos is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = {m: {k: feats(r) for k, r in load(tag).items()} for m, tag in NOTOOL_RUNS.items()}
    ref = data["full"]
    res = {}
    for m, d in data.items():
        ks = sorted(set(d) & set(ref))
        W = sum(d[k]["words"] for k in ks)
        aa = [d[k]["after_answer_share"] for k in ks if d[k]["after_answer_share"] is not None]
        both = [k for k in ks if d[k]["after_answer_share"] is not None and ref[k]["after_answer_share"] is not None]
        res[m] = {
            "attempts_paired": len(ks), "acc": st.mean(d[k]["correct"] for k in ks),
            "tok_mean": st.mean(d[k]["tok"] for k in ks), "ref_tok_mean": st.mean(ref[k]["tok"] for k in ks),
            "tok_ratio_median": st.median(d[k]["tok"] / max(1, ref[k]["tok"]) for k in ks),
            "words_per_attempt": W / len(ks), "length_cut": sum(d[k]["finish"] == "length" for k in ks),
            **{f"{k}_per_1k": 1000 * sum(d[x]["ph"][k] for x in ks) / W for k in PHRASES},
            "let_me_per_1k": 1000 * sum(d[x]["letme"] for x in ks) / W,
            "Need_sentence_per_1k": 1000 * sum(d[x]["need"] for x in ks) / W,
            "rep6_mean": st.mean(x for x in (d[k]["rep6"] for k in ks) if x is not None),
            "after_answer_share_mean": st.mean(aa) if aa else None, "after_answer_n": len(aa),
            "after_answer_share_both_correct": (st.mean(d[k]["after_answer_share"] for k in both),
                                                st.mean(ref[k]["after_answer_share"] for k in both), len(both)),
            "after_answer_tokens_both_correct": (st.mean(d[k]["after_answer_share"] * d[k]["tok"] for k in both),
                                                 st.mean(ref[k]["after_answer_share"] * ref[k]["tok"] for k in both)),
        }
    json.dump(res, open(a.out, "w"), indent=1)
    keys = list(res["full"].keys())
    print("metric".ljust(34) + "".join(m.rjust(12) for m in res))
    for k in keys:
        row = []
        for m in res:
            v = res[m][k]
            row.append(str(tuple(round(x, 3) for x in v)) if isinstance(v, tuple) else
                       (f"{v:12.3f}" if isinstance(v, float) else str(v).rjust(12)))
        print(k.ljust(34) + " ".join(row))


if __name__ == "__main__":
    main()
