#!/usr/bin/env python3
"""E043: teacher-forced replay of tokenized rollouts through the vLLM server with the E043 observer on.

Requests go one at a time (prompt = consumed inputs x[0:N-1], max_tokens=1), so the observer's per-step
records concatenate to exactly the request token streams; reap_aggregate.py checks that via input_ids.
The control file switches observation (and the frozen Engram / expert masks) on for the run and off after.

  replay_reap.py --tok DIR [...] --tokens 1000000 --seed 0 --obs-dir /dev/shm/e043/obs_pilot \
      --ctrl /dev/shm/e043/ctrl.json [--engram-keep masks/x.npz] [--expert-keep keep.npy]
"""
import argparse
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import requests


def post(server, model, ids, salt=True):
    body = {"model": model, "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0.0}
    if salt:
        body["cache_salt"] = uuid.uuid4().hex
    r = requests.post(f"{server}/v1/completions", json=body, timeout=3600)
    r.raise_for_status()
    return r.json()


def write_ctrl(path, **kw):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(kw))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=1_000_000, help="consumed-input tokens to replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=120_000)
    ap.add_argument("--server", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--engram-keep", default=None)
    ap.add_argument("--expert-keep", default=None)
    ap.add_argument("--exclude", default=None, help="manifest of an earlier run whose rollouts to skip")
    a = ap.parse_args()

    recs = []
    for d in a.tok:
        for line in open(Path(d) / "index.jsonl"):
            r = json.loads(line)
            r["dir"] = str(d)
            recs.append(r)
    skip = set()
    if a.exclude:
        skip = {(r["dir"], r["offset"]) for r in map(json.loads, open(a.exclude)) if "offset" in r}
    recs = [r for r in recs if r["length"] - 1 <= a.max_len and (r["dir"], r["offset"]) not in skip]
    random.Random(a.seed).shuffle(recs)
    chosen, n = [], 0
    for r in recs:
        if n >= a.tokens:
            break
        chosen.append(r)
        n += r["length"] - 1

    obs = Path(a.obs_dir)
    obs.mkdir(parents=True, exist_ok=True)
    man = open(obs / "manifest.jsonl", "w")
    toks = {d: np.memmap(Path(d) / "tokens.u32", dtype=np.uint32, mode="r") for d in a.tok}
    warm = [0, 128000, 2000, 3000]  # any short prompt: its step is recorded first and skipped by alignment
    write_ctrl(a.ctrl, observe=str(obs), engram_keep=a.engram_keep, expert_keep=a.expert_keep)
    post(a.server, a.model, warm)
    man.write(json.dumps({"warmup": warm}) + "\n")
    t0, done = time.time(), 0
    for i, r in enumerate(chosen):
        ids = toks[r["dir"]][r["offset"]: r["offset"] + r["length"] - 1]
        post(a.server, a.model, ids)
        man.write(json.dumps(r) + "\n")
        man.flush()
        done += len(ids)
        if (i + 1) % 10 == 0 or i + 1 == len(chosen):
            dt = time.time() - t0
            print(f"{i + 1}/{len(chosen)} rollouts, {done / 1e6:.2f}M tokens, {done / dt:.0f} tok/s", flush=True)
    write_ctrl(a.ctrl, observe=None, engram_keep=a.engram_keep, expert_keep=a.expert_keep)
    post(a.server, a.model, warm)  # flushes the last recorded step
    man.close()
    print(json.dumps({"rollouts": len(chosen), "tokens": done, "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
