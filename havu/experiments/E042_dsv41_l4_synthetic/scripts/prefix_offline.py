#!/usr/bin/env python3
"""E042 RW: offline prefix check on saved harness transcripts (`<run>/*.messages.json.gz`).

For every assistant turn k it renders the prompt the harness sent (messages before that turn, with the generation
prompt) through the server's /tokenize, and checks that prompt k is a token prefix of prompt k+1. A failure there
means the prompt's old part itself changes between turns. It also renders prompt k + assistant k (no generation
prompt) to see where the re-rendered assistant turn sits relative to prompt k+1.

Usage: prefix_offline.py <trace dir> [--base-url URL] [--limit N]
"""
import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as h  # noqa: E402

KW = {"thinking": True, "reasoning_effort": 50}


def tokenize(base, messages, tools, gen):
    body = {"model": "dsv41", "messages": messages, "tools": tools, "chat_template_kwargs": KW,
            "add_generation_prompt": gen, "add_special_tokens": False}
    req = urllib.request.Request(base.rsplit("/v1", 1)[0] + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["tokens"]


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--limit", type=int, default=4)
    a = ap.parse_args()
    args = h.parser().parse_args(["--mode", "grid", "--answer-tool", "--model", "dsv41"])
    tools = h.tools_for(args)
    for f in sorted(Path(a.trace_dir).glob("*.messages.json.gz"))[:a.limit]:
        msgs = json.loads(gzip.open(f).read())
        if isinstance(msgs, dict):
            msgs = msgs.get("messages", msgs)
        idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        prev = None
        for k, i in enumerate(idx):
            p = tokenize(a.base_url, msgs[:i], tools, True)
            if prev is not None:
                d = first_diff(prev, p)
                print(f"{f.name[:24]} turn {k}: prompt {len(p)}; previous prompt {len(prev)} is prefix: "
                      f"{d == len(prev)} (first diff {d})")
            prev = p
        print(f"{f.name[:24]}: {len(idx)} assistant turns checked", flush=True)


if __name__ == "__main__":
    main()
