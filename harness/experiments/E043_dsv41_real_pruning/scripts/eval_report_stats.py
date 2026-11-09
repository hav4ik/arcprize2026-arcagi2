#!/usr/bin/env python3
"""E043: ARC eval statistics from results jsonl + transcripts (small JSON for the report).

  eval_report_stats.py <evals_results_dir> <out.json>
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

EV, OUT = Path(sys.argv[1]), sys.argv[2]
PFX = "tools_persistent_grid_lean2sub_dsv41_low_e043_"
RUNS = {"full": "full_a1", "eng64": "eng64_a1", "E72": "e72_w30_eng64_rescue2", "E64": "e64_w30_eng64_rescue2",
        "E48": "e48_w30_eng64_rescue2", "E64 one-shot (no rescue)": "e64_cdec_eng64", "E72 w10 (no rescue)": "e72_w10_eng64",
        "E64 w30 (no rescue)": "e64_w30_eng64"}
ERR = re.compile(r"Traceback|Error\b|\[tool error")


def main():
    out = {}
    solved = {}
    for name, tag in RUNS.items():
        p = EV / f"{PFX}{tag}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        tw = collections.defaultdict(list)
        n_tool = n_err = n_rescue_code = n_nudge = 0
        toks, turns, calls_per = [], [], []
        for r in rows:
            tw[r["task_id"]].append(bool(r.get("correct")))
            toks.append((r.get("usage") or {}).get("completion", 0))
            turns.append(r.get("turns") or 0)
            tp = r.get("trace_path")
            if tp and Path(tp).exists():
                m = json.load(gzip.open(tp))
                msgs = m["messages"] if isinstance(m, dict) else m
                calls = 0
                for x in msgs:
                    if x["role"] == "tool":
                        n_tool += 1
                        n_err += bool(ERR.search(x.get("content") or ""))
                    if x["role"] == "assistant":
                        for t in x.get("tool_calls") or []:
                            calls += 1
                            n_rescue_code += str(t.get("id", "")).startswith("rescue_")
                    if x["role"] == "user" and "[harness] No valid tool call" in (x.get("content") or ""):
                        n_nudge += 1
                calls_per.append(calls)
        tp_rows = [r for r in rows if r.get("train_pass") is not None]
        tpass = [r for r in tp_rows if r.get("train_pass")]
        out[name] = {
            "tag": tag, "attempts": len(rows), "correct": sum(bool(r.get("correct")) for r in rows),
            "pass1": sum(bool(r.get("correct")) for r in rows) / len(rows),
            "task_weighted": sum(sum(v) / len(v) for v in tw.values()) / len(tw),
            "stops": dict(collections.Counter(r.get("stop_reason") for r in rows)),
            "tokens": toks, "turns": turns, "tool_calls": calls_per,
            "tool_messages": n_tool, "tool_errors": n_err, "tool_error_rate": n_err / max(1, n_tool),
            "rescued_calls": n_rescue_code, "nudges": n_nudge,
            "train_pass": len(tpass), "train_pass_correct": sum(bool(r.get("correct")) for r in tpass),
            "correct_without_train_pass": sum(bool(r.get("correct")) and not r.get("train_pass") for r in rows),
        }
        solved[name] = {(r["task_id"], r["pair"]) for r in rows if r.get("correct")}
        out[name]["pairs"] = sorted(f"{t}:{p}" for t, p in {(r["task_id"], r["pair"]) for r in rows})
    for name in ("eng64", "E72", "E64", "E48"):
        if name in solved:
            pairs = {tuple(x.split(":")) for x in out[name]["pairs"]}
            pairs = {(t, int(p)) for t, p in pairs}
            full = {x for x in solved["full"] if x in pairs}
            out[name]["vs_full"] = {"both": len(full & solved[name]), "full_only": len(full - solved[name]),
                                    "pruned_only": len(solved[name] - full),
                                    "neither": len(pairs) - len(full | solved[name])}
    json.dump(out, open(OUT, "w"))
    for k, v in out.items():
        print(f"{k:28s} {v['correct']}/{v['attempts']} tw={v['task_weighted']:.3f} tool_err={v['tool_error_rate']:.3f} "
              f"rescued={v['rescued_calls']} nudges={v['nudges']} train_pass={v['train_pass']}({v['train_pass_correct']} correct) "
              f"{v.get('vs_full', '')}")


if __name__ == "__main__":
    main()
