#!/usr/bin/env python3
"""E043: is broken free generation a decode-path bug or real model damage?

Greedy-generate N tokens for a chat prompt (token-id completion), then re-score prompt + generation teacher-forced
(prompt_logprobs=1) and check that the prefill argmax at every generated position equals the decoded token.
Mismatches point at the decode path (CUDA graphs, cache, masks applied differently in decode).

  decode_consistency.py --server URL --ctrl FILE --tag T [--engram-keep X] [--expert-keep Y] [--n 200]
"""
import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(__file__))
from tokenize_rollouts import harness_tools  # noqa: E402

MSGS = [{"role": "system", "content": "Reasoning Effort: Minimal. Keep your thinking short."},
        {"role": "user", "content": "What is 17*23? Use the python tool to check."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--ctrl", default="/dev/shm/e043/ctrlB.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--engram-keep", default=None)
    ap.add_argument("--expert-keep", default=None)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.ctrl:
        with open(a.ctrl + ".tmp", "w") as f:
            json.dump({"engram_keep": a.engram_keep and os.path.realpath(a.engram_keep),
                       "expert_keep": a.expert_keep and os.path.realpath(a.expert_keep), "observe": None}, f)
        os.replace(a.ctrl + ".tmp", a.ctrl)
        requests.post(f"{a.server}/v1/completions", json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1})
    tok = requests.post(f"{a.server}/tokenize", json={
        "model": "dsv41", "messages": MSGS, "tools": [harness_tools()[0]],
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}}).json()["tokens"]
    g = requests.post(f"{a.server}/v1/completions", json={
        "model": "dsv41", "prompt": tok, "max_tokens": a.n, "temperature": 0.0, "logprobs": 1,
        "return_token_ids": True}).json()["choices"][0]
    gen = g.get("token_ids")
    if gen is None:
        gen = [int(t.split(":")[-1]) if t.startswith("token_id:") else None for t in g["logprobs"]["tokens"]]
    s = requests.post(f"{a.server}/v1/completions", json={
        "model": "dsv41", "prompt": tok + gen, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}).json()
    plp = s["choices"][0]["prompt_logprobs"]
    mism = []
    for i, t in enumerate(gen):
        d = plp[len(tok) + i]
        best = max(d.items(), key=lambda kv: kv[1]["logprob"])
        if int(best[0]) != t:
            mism.append((i, t, int(best[0]), d.get(str(t), {}).get("logprob"), best[1]["logprob"]))
    print(json.dumps({"tag": a.tag, "generated": len(gen), "argmax_mismatches": len(mism), "first": mism[:5],
                      "text": g["text"][:300]}))
    if a.ctrl:
        with open(a.ctrl + ".tmp", "w") as f:
            json.dump({"engram_keep": None, "expert_keep": None, "observe": None}, f)
        os.replace(a.ctrl + ".tmp", a.ctrl)


if __name__ == "__main__":
    main()
