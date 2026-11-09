#!/usr/bin/env python3
"""Locate a parity failure: compare first-token logprobs over a sequence of runs of one prompt.

  cold c1 (after full reset) -> GPU hit g1 -> GPU hit g2 -> [reset GPU] host restore h1 -> GPU hit g3
  -> [reset GPU] host restore h2 -> [reset all] cold c2

g1 == g2 checks GPU-hit determinism; h1 == h2 host-restore determinism; g3 hits blocks that came back from the host,
so g3 == h1 != g1 means the restored bytes differ from the computed ones; c1 == c2 checks cold determinism.
Stdlib only; needs the server alone (no other traffic).

    python diag_parity.py --len 3000 [--max-tokens 4]
"""
import argparse
import json
import random
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from parity_test import complete, reset, wait_stores_drained  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--len", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--top-logprobs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    prompt = [rng.randrange(1000, 100000) for _ in range(a.len)]
    runs = {}

    def run(name):
        out = complete(a, prompt, a.max_tokens)
        runs[name] = out
        print(f"{name}: cached {out['cached']}, ids {out['ids']}, lp {[round(x, 6) for x in out['token_logprobs']]}",
              flush=True)

    reset(a, external=True)
    run("c1")
    wait_stores_drained(a.url)
    run("g1")
    run("g2")
    reset(a, external=False)
    run("h1")
    wait_stores_drained(a.url)
    run("g3")
    reset(a, external=False)
    run("h2")
    reset(a, external=True)
    run("c2")

    def eq(x, y):
        return runs[x]["ids"] == runs[y]["ids"] and runs[x]["token_logprobs"] == runs[y]["token_logprobs"] \
            and runs[x]["top_logprobs"] == runs[y]["top_logprobs"]

    res = {f"{x}=={y}": eq(x, y) for x, y in
           [("g1", "g2"), ("h1", "h2"), ("g3", "h1"), ("g3", "g1"), ("h1", "g1"), ("c1", "c2"), ("c1", "g1")]}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
