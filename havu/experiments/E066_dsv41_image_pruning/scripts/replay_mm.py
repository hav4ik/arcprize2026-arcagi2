#!/usr/bin/env python3
"""E066: teacher-forced replay of tokenized image rollouts through the vLLM fork with the E043 observer on.

E043's replay_reap.py with chat requests (mm_common.chat_body) instead of token-id prompts, so the vision tower runs
and image positions reach the MoE as image embeddings (router bias_vl). One request at a time, max_tokens=1, a
unique cache_salt; the server consumes the whole stream (manifest "sent": "full"), and class_stats.py aggregate
checks the observer's input_ids against it. --dump-x records MoE inputs for GPTQ calibration as in E043.

  replay_mm.py --tok DIR [...] --parquet e065_image.parquet --tokens 15000000 --seed 1 --obs-dir /dev/shm/e066/obs \
      --ctrl /dev/shm/e043/ctrlB.json --engram-keep engram.npz [--expert-keep keep.npy] [--server URL]
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests

from mm_common import KEY, chat_body, load_index, load_rows

WARM = [0, 128000, 2000, 3000]  # a short token prompt: its step is recorded first and skipped by the alignment


def write_ctrl(path, **kw):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(kw))
    tmp.replace(path)


def warmup(server, model):
    r = requests.post(f"{server}/v1/completions", json={"model": model, "prompt": WARM, "max_tokens": 1,
                                                        "temperature": 0.0}, timeout=600)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=15_000_000, help="consumed tokens to replay")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=120_000)
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--effort", type=int, default=50)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--engram-keep", default=None)
    ap.add_argument("--expert-keep", default=None)
    ap.add_argument("--exclude", default=None, help="manifest of an earlier run whose rollouts to skip")
    ap.add_argument("--dump-x", action="store_true", help="also record every layer's MoE input (GPTQ calibration)")
    ap.add_argument("--dump-x-frac", type=float, default=1.0)
    a = ap.parse_args()

    recs = load_index(a.tok, a.max_len)
    skip = set()
    if a.exclude:
        skip = {(r["dir"], r["offset"]) for r in map(json.loads, open(a.exclude)) if "offset" in r}
    recs = [r for r in recs if (r["dir"], r["offset"]) not in skip]
    random.Random(a.seed).shuffle(recs)
    chosen, n = [], 0
    for r in recs:
        if n >= a.tokens:
            break
        chosen.append(r)
        n += r["length"]
    rows = load_rows(a.parquet)
    missing = [r for r in chosen if tuple(r[k] for k in KEY) not in rows]
    assert not missing, f"{len(missing)} rollouts not in the parquet, e.g. {missing[0]}"

    obs = Path(a.obs_dir)
    obs.mkdir(parents=True, exist_ok=True)
    man = open(obs / "manifest.jsonl", "w")
    write_ctrl(a.ctrl, observe=str(obs), engram_keep=a.engram_keep, expert_keep=a.expert_keep, dump_x=a.dump_x,
               dump_x_frac=a.dump_x_frac)
    warmup(a.server, a.model)
    man.write(json.dumps({"warmup": WARM}) + "\n")
    t0, done = time.time(), 0
    for i, r in enumerate(chosen):
        body = chat_body(rows[tuple(r[k] for k in KEY)], a.model, a.effort)
        resp = requests.post(f"{a.server}/v1/chat/completions", json=body, timeout=3600)
        resp.raise_for_status()
        pt = resp.json()["usage"]["prompt_tokens"]
        if pt != r["length"]:  # the observer check would fail later; stop now with the reason
            raise SystemExit(f"{r['task_id']}: server consumed {pt} prompt tokens, offline stream has {r['length']}")
        man.write(json.dumps({**r, "sent": "full"}) + "\n")
        man.flush()
        done += r["length"]
        if (i + 1) % 10 == 0 or i + 1 == len(chosen):
            dt = time.time() - t0
            print(f"{i + 1}/{len(chosen)} rollouts, {done / 1e6:.2f}M tokens, {done / dt:.0f} tok/s", flush=True)
    write_ctrl(a.ctrl, observe=None, engram_keep=a.engram_keep, expert_keep=a.expert_keep)
    warmup(a.server, a.model)  # flushes the last recorded step
    man.close()
    print(json.dumps({"rollouts": len(chosen), "tokens": done, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
