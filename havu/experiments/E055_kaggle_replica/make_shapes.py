#!/usr/bin/env python3
"""E055: E043 per-turn tool traces (tool_shapes_*.json: per attempt the prompt / completion of every turn) -> the replay
tools format of E042 trace_shapes.py: {"p0": first prompt, "turns": [[completion_i, result_i], ...]} with result_i =
prompt_{i+1} - prompt_i - completion_i (tool output + chat framing), 0 after the last turn.
Usage: make_shapes.py <tool_shapes.json> <out.json>"""
import json, sys
src = json.load(open(sys.argv[1]))["attempts"]
out = []
for a in src:
    t = a["turns"]
    if not t:
        continue
    p = [x["prompt"] for x in t]; c = [x["completion"] for x in t]
    turns = [[max(1, c[i]), max(0, p[i + 1] - p[i] - c[i]) if i + 1 < len(t) else 0] for i in range(len(t))]
    out.append({"p0": p[0], "turns": turns, "task_id": a.get("task_id"), "correct": a.get("correct")})
json.dump(out, open(sys.argv[2], "w"))
size = sorted(s["p0"] + sum(x + y for x, y in s["turns"]) for s in out); n = len(size)
print(f"{n} attempts; final context median {size[n // 2]}, p70 {size[int(.7 * n)]}, p90 {size[int(.9 * n)]}, max {size[-1]}")
