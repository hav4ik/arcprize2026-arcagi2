#!/usr/bin/env python3
"""E043 no-tool round: ARC eval statistics from evals/results/e043nt_<tag>.jsonl (small JSON for the report).

  eval_notool_stats.py <evals_results_dir> <out.json>
Per run: pass@1 over pairs and task-weighted, finish reasons, unparsed answers, output-token distribution, and the
pair-level comparison with the full model (both / full only / pruned only / neither).
"""
import collections
import json
import statistics
import sys
from pathlib import Path

EV, OUT = Path(sys.argv[1]), sys.argv[2]


def tokens(r):
    u = r.get("usage") or {}
    if isinstance(u, str):
        u = json.loads(u.replace("'", '"').replace("None", "null"))
    return u.get("completion_tokens") or 0


def main():
    out, solved = {}, {}
    for p in sorted(EV.glob("e043nt_*.jsonl")):
        tag = p.stem[len("e043nt_"):]
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if "error" not in r]
        if not rows:
            continue
        ok = lambda r: r.get("correct") in (True, "True")
        tw = collections.defaultdict(list)
        for r in rows:
            tw[r["task_id"]].append(ok(r))
        toks = sorted(tokens(r) for r in rows)
        out[tag] = {"attempts": len(rows), "correct": sum(map(ok, rows)), "pass1": sum(map(ok, rows)) / len(rows),
                    "task_weighted": sum(sum(v) / len(v) for v in tw.values()) / len(tw),
                    "finish": dict(collections.Counter(r.get("finish_reason") for r in rows)),
                    "unparsed": sum(r.get("pred") in (None, "", "None") for r in rows),
                    "tokens_mean": statistics.mean(toks), "tokens_median": statistics.median(toks),
                    "tokens_p90": toks[int(0.9 * (len(toks) - 1))], "tokens": toks}
        solved[tag] = {(r["task_id"], str(r["pair"])) for r in rows if ok(r)}
        out[tag]["pairs"] = sorted(f"{r['task_id']}:{r['pair']}" for r in rows)
    if "full_a1" in solved:
        for tag in out:
            if tag == "full_a1":
                continue
            pairs = {tuple(x.split(":")) for x in out[tag]["pairs"]}
            full = {x for x in solved["full_a1"] if x in pairs}
            out[tag]["vs_full"] = {"both": len(full & solved[tag]), "full_only": len(full - solved[tag]),
                                   "pruned_only": len(solved[tag] - full),
                                   "neither": len(pairs) - len(full | solved[tag])}
    json.dump(out, open(OUT, "w"))
    for k, v in out.items():
        print(f"{k:24s} {v['correct']}/{v['attempts']} pass@1={v['pass1']:.3f} tw={v['task_weighted']:.3f} "
              f"finish={v['finish']} unparsed={v['unparsed']} tok mean={v['tokens_mean']:.0f} p90={v['tokens_p90']} "
              f"{v.get('vs_full', '')}")


if __name__ == "__main__":
    main()
