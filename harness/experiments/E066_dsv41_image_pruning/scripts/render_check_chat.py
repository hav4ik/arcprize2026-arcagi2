#!/usr/bin/env python3
"""E066 harness audit: does the chat path re-render a finished turn so that the server's cached prefix survives?

For each assistant turn k of a trace: P_k = /tokenize(messages before turn k, add_generation_prompt) and
P_k+1 likewise. Checks (1) P_k is a prefix of P_k+1 (history stable), and (2) whether turn k's reasoning text is in
the re-rendered history (P_k+1 after P_k), and how many tokens that segment has. Images are dropped from the user
message (same in every render, so prefixes are unaffected). The server must serve the model's chat template.
  render_check_chat.py <trace.messages.json.gz> [...] [--server http://127.0.0.1:18001]
"""
import argparse
import gzip
import json

import requests


def strip_images(m):
    c = m.get("content")
    if isinstance(c, list):
        m = {**m, "content": [p for p in c if p.get("type") == "text"]}
    return m


TOOLS = None


def tok(server, msgs):
    body = {"model": "dsv41", "messages": msgs, "add_generation_prompt": True,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}}
    if TOOLS:
        body["tools"] = TOOLS
    r = requests.post(f"{server}/tokenize", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def detok(server, ids):
    r = requests.post(f"{server}/detokenize", json={"model": "dsv41", "tokens": ids}, timeout=120)
    r.raise_for_status()
    return r.json()["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--tools", default=None, help="JSON file with the harness's tool schema (as the eval requests send)")
    a = ap.parse_args()
    global TOOLS
    TOOLS = json.load(open(a.tools)) if a.tools else None
    for tp in a.traces:
        msgs = [strip_images(m) for m in json.load(gzip.open(tp))["messages"]]
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        print(f"== {tp.split('/')[-1]}: {len(idx)} assistant turns")
        for k in range(len(idx) - 1):
            pk, pn = tok(a.server, msgs[:idx[k]]), tok(a.server, msgs[:idx[k + 1]])
            lcp = next((i for i, (x, y) in enumerate(zip(pk, pn)) if x != y), min(len(pk), len(pn)))
            seg = detok(a.server, pn[len(pk):]) if lcp == len(pk) else ""
            rc = (msgs[idx[k]].get("reasoning_content") or "").strip()
            has = bool(rc) and rc[:60] in seg
            print(f"  turn {k}: P_k {len(pk)} tok, P_k+1 {len(pn)} tok, LCP {lcp} ({'prefix OK' if lcp == len(pk) else 'BROKEN at ' + str(lcp)})"
                  f" | re-rendered turn+tool segment {len(pn) - len(pk)} tok | reasoning chars {len(rc)} | reasoning kept in history: {has}")
            if k == 0:
                print("    segment starts:", json.dumps(seg[:240]))


if __name__ == "__main__":
    main()
