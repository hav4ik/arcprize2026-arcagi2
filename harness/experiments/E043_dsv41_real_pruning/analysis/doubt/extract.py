#!/usr/bin/env python3
"""E043 doubt analysis, step 1: per-attempt / per-turn features of tool-harness traces (CPU only, read-only).

  extract.py --cache DIR [models ...]        (default: every model in common.TOOL_RUNS)

Writes DIR/<model>.pkl: list of attempts {key, correct, stop, comp, train_pass, n_train, pred_len, turns, occ}.
Per turn: exact completion tokens (trace events), tokenizer counts of thinking / code / submit / content,
thinking words, doubt-marker counts, tool-output flags (Python error, train-check verdict), code similarity to
the previous code and to the first passing code, n-gram repetition, and whether the final answer grid is printed.
`occ` lists every core doubt marker with its turn, relative position and the surrounding text.
"""
import argparse
import os
import pickle
import re
import sys
from multiprocessing import Pool

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CORE, MARKER_RE, PYERR, TOOL_RUNS, TOKENIZER, WORD_RE, attempt_turns, core_spans,  # noqa: E402
                    is_train_check, load_records, load_trace, marker_counts, verdict)

_TK = None


def tk():
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(TOKENIZER)
    return _TK


def ntok(s):
    return len(tk().encode(s, add_special_tokens=False).ids) if s else 0


def code_lines(code):
    return {ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")}


def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def norm_sent(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def rep_ngram(words, n=6):
    if len(words) < 50:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def kind_of(text):
    for k in CORE:
        if MARKER_RE[k].fullmatch(text) or MARKER_RE[k].search(text):
            return k
    return "other"


def features(args):
    key, r, keep_text = args
    try:
        turns, ntr = attempt_turns(load_trace(r["trace_path"]))
    except Exception as e:  # noqa: BLE001
        return {"key": key, "bad": repr(e)}
    pred = r.get("pred")
    pdig = ""
    if isinstance(pred, list) and pred and all(isinstance(row, list) for row in pred):
        pdig = "".join(str(c) for row in pred for c in row)
    seen_sent = set()
    first_pass_code = None
    prev_code = None
    out = []
    occ = []
    for i, t in enumerate(turns):
        th = t["think"]
        words = WORD_RE.findall(th)
        sents = [norm_sent(s) for s in SENT_SPLIT.split(th)]
        sents = [s for s in sents if len(s.split()) >= 6]
        xrep_words = sum(len(s.split()) for s in sents if s in seen_sent)
        seen_sent.update(sents)
        v = verdict(t["code"], t["out"], ntr)
        cl = code_lines(t["code"])
        feat = {
            "comp": t["comp"], "finish": t["finish"], "names": t["names"],
            "think_tok": ntok(th), "code_tok": ntok(t["code"]), "sub_tok": ntok(t["submit"]),
            "content_tok": ntok(t["content"]), "words": len(words),
            "mk": marker_counts(th), "core": len(core_spans(th)),
            "err": bool(t["out"] and PYERR.search(t["out"])), "check": bool(t["code"] and is_train_check(t["code"])),
            "verdict": v, "has_code": bool(t["code"].strip()),
            "sim_prev": jacc(cl, prev_code) if prev_code is not None and cl else None,
            "sim_pass": jacc(cl, first_pass_code) if first_pass_code is not None and cl else None,
            "rep6": rep_ngram(words), "xrep": xrep_words / max(1, len(words)),
            "pred_in_out": bool(len(pdig) >= 12 and pdig in re.sub(r"\D", "", t["out"])),
            "out_len": len(t["out"]),
        }
        if keep_text:
            feat["think_head"] = th[:300]
            feat["code_head"] = t["code"][:300]
            feat["out_head"] = t["out"][:300]
        out.append(feat)
        if cl:
            prev_code = cl
        if v == "pass" and first_pass_code is None:
            first_pass_code = cl
        for s, e, txt in core_spans(th):
            occ.append({"turn": i, "kind": kind_of(txt), "txt": txt, "pos": s / max(1, len(th)),
                        "chars_after": len(th) - e, "pre": th[max(0, s - 100):s], "post": th[e:e + 240],
                        "tok_after": ntok(th[e:]) if keep_text else None})
    return {"key": key, "correct": bool(r.get("correct")), "stop": r.get("stop_reason"),
            "comp": (r.get("usage") or {}).get("completion", 0), "train_pass": r.get("train_pass"),
            "n_train": ntr, "pred_len": len(pdig), "turns": out, "occ": occ, "trace_path": r["trace_path"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    for m in a.models or list(TOOL_RUNS):
        recs = load_records(m)
        jobs = [(k, r, True) for k, r in sorted(recs.items())]
        with Pool(a.procs) as p:
            res = p.map(features, jobs, chunksize=2)
        bad = [x for x in res if "bad" in x]
        res = [x for x in res if "bad" not in x]
        pickle.dump(res, open(os.path.join(a.cache, f"{m}.pkl"), "wb"))
        print(m, len(res), "attempts", len(bad), "bad", flush=True)


if __name__ == "__main__":
    main()
