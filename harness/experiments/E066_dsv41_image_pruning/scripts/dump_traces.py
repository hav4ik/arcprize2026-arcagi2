#!/usr/bin/env python3
"""E066 trace review: readable text dumps of a run's finished attempts (reasoning, tool calls, tool outputs; images
dropped, long tool outputs cut), one file per (task, attempt), plus index.tsv (task, attempt, pairs, correct,
turns, output tokens, tool errors, rescues, stop reason).

  dump_traces.py <run suffix> <out dir> [--tasks t1,t2,...] [--max-tool-chars 1500]
Run suffixes: dsv41_low_e066_e72_img (E72-152), dsv41_low_e066_c1x_img (candidate 1),
deepseek-flash_low_e058_pilot30 (full model, API), dsv41_low_e066_full_ref (full model, self-hosted).
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_"


def text_of(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") if p.get("type") == "text" else "[image]" for p in c)
    return c or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--max-tool-chars", type=int, default=1500)
    a = ap.parse_args()
    want = set(a.tasks.split(",")) if a.tasks else None
    latest = {}
    for l in open(R + a.run + ".jsonl"):
        r = json.loads(l)
        latest[(r["task_id"], r["pair"], r["attempt"])] = r
    att = defaultdict(list)
    for (t, p, at), r in latest.items():
        if at <= 1 and (want is None or t in want):
            att[(t, at)].append(r)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for (t, at), rs in sorted(att.items()):
        if any("error" in r for r in rs):
            continue
        rs.sort(key=lambda r: r["pair"])
        r0 = rs[0]
        u = r0.get("usage") or {}
        rows.append([t, at, len(rs), sum(bool(r.get("correct")) for r in rs), r0.get("turns"), (u.get("completion_tokens") or u.get("completion")),
                     r0.get("tool_errors"), r0.get("rescues"), r0.get("stop_reason")])
        tp = r0.get("trace_path")
        lines = [f"# task {t} attempt {at} | pairs correct {rows[-1][3]}/{len(rs)} | turns {r0.get('turns')} | "
                 f"output tokens {u.get('completion_tokens')} | tool errors {r0.get('tool_errors')} | rescues {r0.get('rescues')} | "
                 f"stop {r0.get('stop_reason')} | answer source {r0.get('answer_source')}"]
        if tp and os.path.exists(tp):
            d = json.load(gzip.open(tp)) if tp.endswith(".gz") else json.load(open(tp))
            for i, m in enumerate(d.get("messages", [])):
                role = m.get("role")
                if role == "system":
                    continue
                if role == "user" and i <= 2:
                    lines.append(f"\n## user (task prompt, {len(text_of(m.get('content')))} chars; images shown as [image])\n" + text_of(m.get("content"))[:6000])
                    continue
                rc = m.get("reasoning_content") or m.get("reasoning")
                if rc:
                    lines.append(f"\n## {role} reasoning\n{rc}")
                c = text_of(m.get("content"))
                if c.strip():
                    cc = c if role != "tool" or len(c) <= a.max_tool_chars else c[:a.max_tool_chars] + f"\n...[{len(c) - a.max_tool_chars} chars cut]"
                    lines.append(f"\n## {role}\n{cc}")
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        code = json.loads(fn.get("arguments") or "{}").get("code", fn.get("arguments"))
                    except json.JSONDecodeError:
                        code = "[unparseable arguments] " + str(fn.get("arguments"))[:2000]
                    lines.append(f"\n## tool call {fn.get('name')}\n{code}")
        else:
            lines.append("\n[no trace file]")
        open(os.path.join(a.out, f"{t}_a{at}.txt"), "w").write("\n".join(lines))
    with open(os.path.join(a.out, "index.tsv"), "w") as f:
        f.write("task\tattempt\tpairs\tcorrect\tturns\toutput_tokens\ttool_errors\trescues\tstop\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(json.dumps({"run": a.run, "attempts": len(rows), "out": a.out}))


if __name__ == "__main__":
    main()
