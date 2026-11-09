#!/usr/bin/env python3
"""E043: per-attempt, per-turn traffic shapes of a tool-harness run, for E042's 4xL4 multi-turn replays.

From each attempt's trace (`events` with per-turn usage), one record per attempt:
  {"task_id", "pair", "attempt", "correct", "stop_reason", "seconds",
   "turns": [{"prompt": P, "cache_hit": C, "completion": N, "reasoning": R, "finish": F}, ...],
   "tool_output_chars": [len of each tool message, in order]}
plus a summary (turns, prompt p50/p90, completion per turn, tool output chars). cache_hit is 0 for self-hosted
runs (vLLM does not report cached tokens per request); use the server's prefix-cache hit rate instead.

  tool_turn_shapes.py e043_e128t_w10_u2_gptq_eng64_sub80 --out results/tool_shapes_e128t.json
"""
import argparse
import glob
import gzip
import json
import os
import statistics

RES = os.path.join(os.path.dirname(__file__), "../../../evals/results")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (path,) = glob.glob(os.path.join(RES, f"tools_*_{a.tag}.jsonl"))
    attempts = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("trace_path") or not os.path.exists(r["trace_path"]):
            continue
        tr = json.load(gzip.open(r["trace_path"]))
        turns = []
        for e in tr.get("events", []):
            u = e.get("usage")
            if not u:
                continue
            det = u.get("prompt_tokens_details") or {}
            turns.append({"prompt": u.get("prompt_tokens", 0),
                          "cache_hit": det.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0,
                          "completion": u.get("completion_tokens", 0),
                          "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                          "finish": e.get("finish_reason")})
        tools = [len(m.get("content") or "") for m in tr["messages"] if m.get("role") == "tool"]
        attempts.append({k: r.get(k) for k in ("task_id", "pair", "attempt", "correct", "stop_reason", "seconds")}
                        | {"turns": turns, "tool_output_chars": tools})
    prompts = [t["prompt"] for x in attempts for t in x["turns"]]
    comps = [t["completion"] for x in attempts for t in x["turns"]]
    tools = [c for x in attempts for c in x["tool_output_chars"]]
    summary = {
        "tag": a.tag, "attempts": len(attempts),
        "turns_mean": round(statistics.mean(len(x["turns"]) for x in attempts), 2),
        "turns_p90": pct([len(x["turns"]) for x in attempts], 0.9),
        "turn_prompt_p50": pct(prompts, 0.5), "turn_prompt_p90": pct(prompts, 0.9),
        "turn_completion_mean": round(statistics.mean(comps)), "turn_completion_p50": pct(comps, 0.5),
        "turn_completion_p90": pct(comps, 0.9),
        "attempt_completion_mean": round(statistics.mean(sum(t["completion"] for t in x["turns"]) for x in attempts)),
        "tool_output_chars_p50": pct(tools, 0.5), "tool_output_chars_p90": pct(tools, 0.9),
    }
    json.dump({"summary": summary, "attempts": attempts}, open(a.out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
