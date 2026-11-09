"""Index E009/E011 sub80s40 + sub96n attempts: per-attempt trace stats (no API calls).

Writes experiments/E017_reasoning_efficiency/attempts_index.jsonl
"""
import gzip, json, re, glob, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
OUT = ROOT / "experiments/E017_reasoning_efficiency"
RUNS = {"sub80s40": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
        "sub96n": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n.jsonl"}


def solved_by_any():
    solved, seen = collections.defaultdict(bool), set()
    for f in glob.glob(str(RES / "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task_id" not in r or "correct" not in r:
                continue
            k = (r["task_id"], r.get("pair", 0))
            seen.add(k)
            solved[k] |= bool(r["correct"])
    return solved, seen


def load(run):
    recs = {}
    for line in open(RES / RUNS[run]):
        r = json.loads(line)
        k = (r["task_id"], r["pair"], r["attempt"])
        if "error" in r and k in recs:
            continue
        recs[k] = r  # last record wins unless it is an error after a good one
    return [r for r in recs.values() if "error" not in r]


def call_code(c):
    a = c["function"]["arguments"]
    try:
        v = json.loads(a)
        return v.get("code", "") if isinstance(v, dict) else str(v)
    except json.JSONDecodeError:
        return a


DIGIT_LINE = re.compile(r"^[\s\[\],0-9.|-]*\d[\s\[\],0-9.|-]*$")


def trace_stats(path):
    d = json.load(gzip.open(path))
    msgs, events = d["messages"], d["events"]
    usage = [e for e in events if "usage" in e]
    turns = []
    ai = 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        u = usage[ai]["usage"] if ai < len(usage) else {}
        ai += 1
        rc = m.get("reasoning_content") or ""
        code = "".join(call_code(c) for c in m.get("tool_calls", []) or [])
        lines = rc.splitlines()
        grid_lines = sum(1 for l in lines if len(l.strip()) >= 7 and DIGIT_LINE.match(l) and sum(ch.isdigit() for ch in l) >= 5)
        turns.append({"tok": u.get("completion_tokens", 0), "rtok": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                      "rchars": len(rc), "codechars": len(code), "contentchars": len(m.get("content") or ""),
                      "grid_lines": grid_lines, "lines": len(lines),
                      "tools": [c["function"]["name"] for c in m.get("tool_calls", []) or []]})
    tool_out = [m["content"] for m in msgs if m["role"] == "tool"]
    errs = sum(1 for t in tool_out if "Traceback" in t or "Error" in t[:3000] and "[completed]" not in t[:20])
    return turns, tool_out, errs


def main():
    solved, seen = solved_by_any()
    rows = []
    for run in RUNS:
        for r in load(run):
            turns, tool_out, errs = trace_stats(r["trace_path"])
            tok = r["usage"]["completion"]
            if r.get("correct"):
                outcome = "correct"
            elif r.get("pred") is not None:
                outcome = "wrong"
            else:
                outcome = "none"
            rows.append({"run": run, "task_id": r["task_id"], "pair": r["pair"], "attempt": r["attempt"],
                         "outcome": outcome, "tokens": tok, "turns": r["turns"], "tool_calls": r["tool_calls"],
                         "tool_errors": r["tool_errors"], "stop": r["stop_reason"], "answer_source": r.get("answer_source"),
                         "pair_solved_any": solved[(r["task_id"], r["pair"])],
                         "reasoning_chars": sum(t["rchars"] for t in turns), "code_chars": sum(t["codechars"] for t in turns),
                         "grid_lines": sum(t["grid_lines"] for t in turns), "reason_lines": sum(t["lines"] for t in turns),
                         "max_turn_tok": max((t["tok"] for t in turns), default=0),
                         "turn_tok": [t["tok"] for t in turns], "trace_errs": errs,
                         "trace_path": r["trace_path"]})
    with open(OUT / "attempts_index.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(len(rows), collections.Counter((r["run"], r["outcome"]) for r in rows))
    pairs = collections.defaultdict(list)
    for r in rows:
        pairs[(r["task_id"], r["pair"])].append(r["outcome"])
    print("pairs", len(pairs), "never solved by any config:", sum(not solved[k] for k in pairs))


main()
