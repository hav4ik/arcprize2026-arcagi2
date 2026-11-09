#!/usr/bin/env python3
"""E043: no-tool free-generation probe on DEV-split training prompts (never the eval set).

For each config (tag:engram_keep|none:expert_keep|none) it writes the control file and sends the first two messages
(effort header + task prompt with summary) of N dev rollouts, greedy and at T=1. Reports: finished (not cut by
max_tokens), answer parses as a grid, and agreement with the original (teacher) answer where the teacher was correct.

  gen_probe_nt.py --dev-index data/tok_nt_dev/index.jsonl --parquets <notool parquets> --n 8 --samples 1 \
      --out probes.jsonl tag:eng:exp [...]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

sys.path.insert(0, "/workspace/dsv4/code/arcprize")
from arcprize_parse import parse  # noqa: E402


def chat(server, messages, max_tokens, temperature, salt):
    body = {"model": "dsv41", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "cache_salt": salt}
    r = requests.post(f"{server}/v1/chat/completions", json=body, timeout=3600).json()
    c = r["choices"][0]
    return c["finish_reason"], c["message"].get("content") or "", r["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--ctrl", default="/dev/shm/e043/ctrlB.json")
    ap.add_argument("--dev-index", required=True)
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--out", required=True)
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    dev = [json.loads(l) for l in open(a.dev_index)]
    want = {(r["task_id"], int(r["pair"])) for r in dev[: a.n]}
    rows = []
    for p in a.parquets:
        for r in pq.read_table(p, columns=["task_id", "pair", "correct", "pred", "messages"]).to_pylist():
            if (r["task_id"], int(r["pair"])) in want:
                rows.append(r)
    rows = rows[: a.n]
    for spec in a.configs:
        tag, eng, exp = spec.split(":")
        ctrl = {"engram_keep": None if eng == "none" else os.path.realpath(eng),
                "expert_keep": None if exp == "none" else os.path.realpath(exp), "observe": None}
        with open(a.ctrl + ".tmp", "w") as f:
            json.dump(ctrl, f)
        os.replace(a.ctrl + ".tmp", a.ctrl)
        requests.post(f"{a.server}/v1/completions", json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1})
        salt = f"{tag}-{time.time()}"
        jobs = [(r, t) for r in rows for t in [0.0] + [1.0] * a.samples]

        def run(job):
            r, t = job
            msgs = json.loads(r["messages"])[:2]
            fin, content, ntok = chat(a.server, msgs, a.max_tokens, t, salt)
            pred = parse(content)
            teacher = json.loads(r["pred"]) if isinstance(r["pred"], str) and r["pred"].startswith("[") else r["pred"]
            return {"config": tag, "task_id": r["task_id"], "temperature": t, "finish": fin, "tokens": ntok,
                    "parsed": pred is not None, "teacher_correct": bool(r["correct"]),
                    "agrees_with_teacher": bool(r["correct"]) and pred == teacher}
        t0 = time.time()
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(run, jobs))
        with open(a.out, "a") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        n = len(res)
        tc = [x for x in res if x["teacher_correct"]]
        print(f"== {tag} ({time.time() - t0:.0f}s): finished {sum(x['finish'] == 'stop' for x in res)}/{n}, "
              f"parsed {sum(x['parsed'] for x in res)}/{n}, agrees with correct teacher "
              f"{sum(x['agrees_with_teacher'] for x in tc)}/{len(tc)}, mean tokens {sum(x['tokens'] for x in res) / n:.0f}",
              flush=True)
    with open(a.ctrl + ".tmp", "w") as f:
        json.dump({"engram_keep": None, "expert_keep": None, "observe": None}, f)
    os.replace(a.ctrl + ".tmp", a.ctrl)


if __name__ == "__main__":
    main()
