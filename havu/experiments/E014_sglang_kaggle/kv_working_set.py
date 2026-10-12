"""Offline estimate of the live KV working set of C concurrent tool agents (E014).

Stationary distribution of one agent's context: time ~ generated tokens, so pick a generated token uniformly over
all turns of all replay attempts; its context = turn start context + position in the turn. The working set of C
desynchronised agents is the sum of C independent draws (Monte Carlo). Compares quantiles with KV capacities.

    .venv/bin/python experiments/E014_sglang_kaggle/kv_working_set.py
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data = json.load(gzip.open(os.path.join(ROOT, "kaggle/datasets/agentic_replay/agentic_replay.json.gz"), "rt"))
starts, lens = [], []
for x in data:
    for t in x["turns"]:
        if t["gen"] > 0:
            starts.append(t["ctx"])
            lens.append(t["gen"])
starts, lens = np.array(starts, float), np.array(lens, float)
p = lens / lens.sum()
rng = np.random.default_rng(0)
N = 20000


def draw(n):
    i = rng.choice(len(p), size=n, p=p)
    return starts[i] + rng.random(n) * lens[i]


one = draw(200000)
print(f"per-agent live context: mean {one.mean():.0f}, p50 {np.median(one):.0f}, p90 {np.quantile(one, .9):.0f}, "
      f"p99 {np.quantile(one, .99):.0f}")
caps = {"vLLM 0.23 (1.40M)": 1.40e6, "SGLang radix default (772k)": 7.72e5}
caps.update({k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])})
rows = []
for c in (16, 20, 24, 28, 32, 35, 40, 48):
    s = draw(N * c).reshape(N, c).sum(1)
    q = {k: float(np.quantile(s, k)) for k in (0.5, 0.9, 0.99, 0.999)}
    over = {name: float((s > cap).mean()) for name, cap in caps.items()}
    rows.append({"concurrency": c, "mean": float(s.mean()), **{f"p{int(k * 1000) / 10:g}": v for k, v in q.items()},
                 "P(over cap)": over})
    print(c, f"mean {s.mean() / 1e3:.0f}k p50 {q[0.5] / 1e3:.0f}k p90 {q[0.9] / 1e3:.0f}k p99 {q[0.99] / 1e3:.0f}k "
          f"p99.9 {q[0.999] / 1e3:.0f}k", {k: round(v, 3) for k, v in over.items()})
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "kv_working_set.json"), "w"), indent=1)
