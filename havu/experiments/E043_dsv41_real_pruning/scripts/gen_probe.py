#!/usr/bin/env python3
"""E043: free-generation probe of mask configurations on a server with the E043 hooks (temperature 0).

For each config (tag:engram_keep|none:expert_keep|none) it writes the control file, then sends a trivial tool
prompt and the first turn of N real ARC eval prompts (system + user from the E030 s20x60_hdrmin traces, harness
tools). Reports whether a valid tool call (run_python / submit_answer with a `code` argument) comes back, the
number of calls, and the start of the reasoning.

  gen_probe.py --server URL --ctrl FILE --parquet eval.parquet --n 3 --out probes.jsonl tag:eng:exp [...]
"""
import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(__file__))
from tokenize_rollouts import harness_tools  # noqa: E402

SIMPLE = [{"role": "system", "content": "Reasoning Effort: Minimal. Keep your thinking short."},
          {"role": "user", "content": "What is 17*23? Use the python tool to check."}]
VALID = {"run_python", "submit_answer"}


def chat(server, messages, tools, max_tokens, temperature=0.0, salt=None):
    body = {"model": "dsv41", "messages": messages, "tools": tools, "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}}
    if salt:  # never reuse KV cached under another mask configuration
        body["cache_salt"] = salt
    r = requests.post(f"{server}/v1/chat/completions", json=body, timeout=1800)
    r.raise_for_status()
    c = r.json()["choices"][0]
    m = c["message"]
    calls = m.get("tool_calls") or []
    ok = bool(calls) and all(t["function"]["name"] in VALID and "code" in json.loads(t["function"]["arguments"] or "{}")
                             for t in calls)
    return {"finish": c["finish_reason"], "n_calls": len(calls), "valid": ok,
            "names": sorted({t["function"]["name"] for t in calls}),
            "reasoning": (m.get("reasoning_content") or m.get("reasoning") or "")[:240], "content": (m.get("content") or "")[:160],
            "completion_tokens": r.json()["usage"]["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--ctrl", default="/dev/shm/e043/ctrlB.json")
    ap.add_argument("--parquet", default="/workspace/dsv4/data/e030_s20x60_hdrmin_eval.parquet")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--samples", type=int, default=0, help="extra temperature-1.0 samples per ARC prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    rows = [r for r in pq.read_table(a.parquet, columns=["task_id", "messages"]).to_pylist()][: a.n]
    arc = [(r["task_id"], json.loads(r["messages"])[:2]) for r in rows]
    tools = harness_tools()
    for spec in a.configs:
        tag, eng, exp = spec.split(":")
        ctrl = {"engram_keep": None if eng == "none" else os.path.realpath(eng),
                "expert_keep": None if exp == "none" else os.path.realpath(exp), "observe": None}
        with open(a.ctrl + ".tmp", "w") as f:
            json.dump(ctrl, f)
        os.replace(a.ctrl + ".tmp", a.ctrl)
        requests.post(f"{a.server}/v1/completions", json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1},
                      timeout=120)
        t0 = time.time()
        salt = f"{tag}-{time.time()}"
        res = [("simple", chat(a.server, SIMPLE, [tools[0]], 400, salt=salt))]
        res += [(tid, chat(a.server, msgs, tools, a.max_tokens, salt=salt)) for tid, msgs in arc]
        res += [(f"{tid}@T1#{k}", chat(a.server, msgs, tools, a.max_tokens, 1.0, salt))
                for tid, msgs in arc for k in range(a.samples)]
        with open(a.out, "a") as f:
            for name, r in res:
                f.write(json.dumps({"config": tag, "probe": name, **r}) + "\n")
        print(f"== {tag} ({time.time() - t0:.0f}s): valid {sum(r['valid'] for _, r in res)}/{len(res)}", flush=True)
        for name, r in res:
            print(f"   {name}: finish={r['finish']} calls={r['n_calls']} valid={r['valid']} names={r['names']} "
                  f"tok={r['completion_tokens']} | {r['reasoning'][:150]!r}", flush=True)
    with open(a.ctrl + ".tmp", "w") as f:
        json.dump({"engram_keep": None, "expert_keep": None, "observe": None}, f)
    os.replace(a.ctrl + ".tmp", a.ctrl)


if __name__ == "__main__":
    main()
