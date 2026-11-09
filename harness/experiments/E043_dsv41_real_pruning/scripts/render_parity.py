#!/usr/bin/env python3
"""E043 / E042: does the next turn's prompt reproduce the tokens the server actually sampled? (prefix-cache reuse)

For N tool dev prompts: stream one turn exactly like evals/run_deepseek_tools.py (temperature 1.0, harness tools,
effort 50, minimal header as sent), collecting the sampled token ids. Rebuild the assistant message as the harness
does ({content, reasoning_content, tool_calls} from the stream deltas), append a tool result per call, render the next
prompt with the server's /tokenize, and compare it with prompt_ids + sampled_ids. Variants of the returned assistant
message: `reasoning_content` (what the harness sends), `reasoning` (the field this vLLM returns).

  render_parity.py --server http://127.0.0.1:18001 --n 12 --out render_parity.jsonl
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(__file__))
from tokenize_rollouts import harness_tools  # noqa: E402

KW = {"thinking": True, "reasoning_effort": 50}


def stream_turn(server, messages, tools, max_tokens, salt):
    body = {"model": "dsv41", "messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": 1.0,
            "top_p": 1.0, "chat_template_kwargs": KW, "cache_salt": salt, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True}
    content, reasoning, calls, ids, prompt_ids, finish = [], [], {}, [], None, None
    with requests.post(f"{server}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            c = json.loads(line[6:])
            prompt_ids = c.get("prompt_token_ids") or prompt_ids
            for ch in c.get("choices") or []:
                ids += ch.get("token_ids") or []
                d = ch.get("delta") or {}
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
                for tc in d.get("tool_calls") or []:
                    call = calls.setdefault(tc["index"], {"id": "", "type": "function",
                                                          "function": {"name": "", "arguments": ""}})
                    call["id"] += tc.get("id") or ""
                    f = tc.get("function") or {}
                    call["function"]["name"] += f.get("name") or ""
                    call["function"]["arguments"] += f.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, prompt_ids, ids, finish


def render(server, messages, tools):
    r = requests.post(f"{server}/tokenize", json={"model": "dsv41", "messages": messages, "tools": tools,
                                                    "chat_template_kwargs": KW, "add_generation_prompt": True},
                      timeout=600)
    r.raise_for_status()
    return r.json()["tokens"]


def detok(server, ids):
    r = requests.post(f"{server}/detokenize", json={"model": "dsv41", "tokens": ids}, timeout=600)
    return r.json().get("prompt", "")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def one(server, tools, k, msgs, max_tokens):
    msg, pids, gids, finish = stream_turn(server, msgs, tools, max_tokens, f"rp-{time.time()}-{k}")
    out = {"probe": k, "finish": finish, "prompt": len(pids or []), "gen": len(gids), "calls": len(msg.get("tool_calls") or [])}
    if not msg.get("tool_calls"):
        return out
    tool_msgs = [{"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in msg["tool_calls"]]
    want = pids + gids
    for variant in ("reasoning_content", "reasoning"):
        m = dict(msg)
        if variant == "reasoning":
            m["reasoning"] = m.pop("reasoning_content")
        nxt = render(server, msgs + [m] + tool_msgs, tools)
        i = first_diff(nxt, want)
        res = {"prompt_prefix_ok": nxt[:len(pids)] == pids, "match": i, "gen_offset": i - len(pids),
               "gen_frac": round((i - len(pids)) / max(1, len(gids)), 3), "full_turn_match": i >= len(want)}
        if i < len(want):
            res["want_text"] = detok(server, want[i:i + 12])
            res["got_text"] = detok(server, nxt[i:i + 12])
        out[variant] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--parquet", default="/workspace/dsv4/data/tool_dev_probe.parquet")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = pq.read_table(a.parquet, columns=["messages"]).to_pylist()
    prompts = [json.loads(r["messages"])[:2] for r in rows]
    jobs = [(k, prompts[k % len(prompts)]) for k in range(a.n)]
    tools = harness_tools()
    with ThreadPoolExecutor(a.workers) as pool:
        outs = list(pool.map(lambda j: one(a.server, tools, j[0], j[1], a.max_tokens), jobs))
    with open(a.out, "a") as f:
        for o in outs:
            f.write(json.dumps(o) + "\n")
            print(json.dumps(o)[:600])


if __name__ == "__main__":
    main()
