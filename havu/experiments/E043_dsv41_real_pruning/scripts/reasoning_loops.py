#!/usr/bin/env python3
"""E043: is a model reasoning longer, and is it looping? Compares output tokens on the same (task, pair, attempt) set
and the gzip compression ratio (compressed / raw bytes) of each reasoning trace, overall and on its last 20K chars
(degenerate loops compress far better: low ratio).

  reasoning_loops.py <evals_results_dir> <tag> [<tag> ...] [--ref full_a1] [--json out.json]
"""
import argparse
import gzip
import json
import statistics
import zlib
from pathlib import Path

TAIL = 20000


def ratio(b):
    return len(zlib.compress(b, 6)) / max(1, len(b))


def load(ev, tag):
    out = {}
    for line in open(ev / f"e043nt_{tag}.jsonl"):
        r = json.loads(line)
        if "error" in r:
            continue
        u = r.get("usage") or {}
        if isinstance(u, str):
            u = json.loads(u.replace("'", '"').replace("None", "null"))
        text = b""
        p = r.get("reasoning_path")
        if p:
            fp = ev.parent.parent / p if not Path(p).is_absolute() else Path(p)
            if fp.exists():
                text = gzip.open(fp, "rb").read()
        out[(r["task_id"], str(r["pair"]), str(r["attempt"]))] = {
            "tokens": u.get("completion_tokens") or 0, "correct": r.get("correct") in (True, "True"),
            "finish": r.get("finish_reason"), "gz": ratio(text) if text else None,
            "gz_tail": ratio(text[-TAIL:]) if len(text) > TAIL else None, "chars": len(text)}
    return out


def summary(rows):
    tok = [r["tokens"] for r in rows]
    gz = [r["gz"] for r in rows if r["gz"] is not None]
    tail = [r["gz_tail"] for r in rows if r["gz_tail"] is not None]
    loops = [r for r in rows if r["gz_tail"] is not None and r["gz_tail"] < 0.15]
    return {"n": len(rows), "tokens_mean": statistics.mean(tok), "tokens_median": statistics.median(tok),
            "gz_mean": statistics.mean(gz) if gz else None, "gz_tail_mean": statistics.mean(tail) if tail else None,
            "loops_tail_lt_0.15": len(loops), "loops_correct": sum(r["correct"] for r in loops),
            "length_cut": sum(r["finish"] == "length" for r in rows), "acc": sum(r["correct"] for r in rows) / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default="full_a1")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev = Path(a.evals)
    ref = load(ev, a.ref)
    res = {}
    for tag in a.tags:
        d = load(ev, tag)
        keys = [k for k in d if k in ref]
        long_keys = [k for k in keys if max(d[k]["tokens"], ref[k]["tokens"]) > 50000]
        res[tag] = {"same_keys": summary([d[k] for k in keys]), "ref_same_keys": summary([ref[k] for k in keys]),
                    "long_same_keys": summary([d[k] for k in long_keys]) if long_keys else None,
                    "ref_long_same_keys": summary([ref[k] for k in long_keys]) if long_keys else None,
                    "tokens_ratio_per_attempt_median": statistics.median(d[k]["tokens"] / max(1, ref[k]["tokens"]) for k in keys),
                    "longer_than_ref": sum(d[k]["tokens"] > ref[k]["tokens"] for k in keys)}
        s, rs = res[tag]["same_keys"], res[tag]["ref_same_keys"]
        print(f"== {tag} vs {a.ref} on the same {s['n']} attempts:")
        for name, x in ((tag, s), (a.ref, rs)):
            print(f"   {name:24s} tokens mean {x['tokens_mean']/1000:5.1f}K median {x['tokens_median']/1000:5.1f}K | gzip ratio "
                  f"{x['gz_mean']:.3f}, last-20K {x['gz_tail_mean']:.3f} | tail<0.15 (loop-like): {x['loops_tail_lt_0.15']} "
                  f"({x['loops_correct']} correct) | length cut {x['length_cut']} | acc {100*x['acc']:.1f}%")
        print(f"   per-attempt token ratio median {res[tag]['tokens_ratio_per_attempt_median']:.2f}; longer than {a.ref} in "
              f"{res[tag]['longer_than_ref']}/{s['n']}")
        if long_keys:
            l, rl = res[tag]["long_same_keys"], res[tag]["ref_long_same_keys"]
            print(f"   long attempts (>50K in either, n={l['n']}): gzip last-20K {tag} {l['gz_tail_mean']:.3f} vs {a.ref} "
                  f"{rl['gz_tail_mean']:.3f}; loop-like {l['loops_tail_lt_0.15']} vs {rl['loops_tail_lt_0.15']}; acc "
                  f"{100*l['acc']:.1f}% vs {100*rl['acc']:.1f}%")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
