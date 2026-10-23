#!/usr/bin/env python3
"""E043: teacher-forced NLL of held-out traces (cheap pruning-damage proxy).

Sends each sequence x[0:N] as a prompt with prompt_logprobs=0 and max_tokens=1 and scores
log p(x[t] | x[<t]). Reports mean NLL and top-1 agreement over model-generated tokens (role 1), separately
for all tokens, per sequence, so candidate masks can be compared paired against the full model.

  nll_eval.py --tok DIR --n-seq 40 --seed 0 --tag full --out results.jsonl [--server URL] [--workers 4]
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests


def score(server, model, ids, roles):
    body = {"model": model, "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0}
    r = requests.post(f"{server}/v1/completions", json=body, timeout=7200)
    r.raise_for_status()
    plp = r.json()["choices"][0]["prompt_logprobs"]
    lp = np.zeros(len(ids))
    rank = np.zeros(len(ids), dtype=np.int64)
    for t in range(1, len(ids)):
        d = plp[t]
        e = d.get(str(int(ids[t]))) or d.get(int(ids[t]))
        lp[t] = e["logprob"]
        rank[t] = e.get("rank") or 0
    g = roles[1:] == 1
    nll = -lp[1:]
    return {"n_gen": int(g.sum()), "nll_gen": float(nll[g].sum()), "top1_gen": int((rank[1:][g] == 1).sum()),
            "n_all": len(ids) - 1, "nll_all": float(nll.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=60000)
    ap.add_argument("--server", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.tok)
    toks = np.fromfile(d / "tokens.u32", dtype=np.uint32)
    roles = np.fromfile(d / "roles.u8", dtype=np.uint8)
    recs = [json.loads(l) for l in open(d / "index.jsonl")]
    recs = [r for r in recs if r["length"] <= a.max_len]
    random.Random(a.seed).shuffle(recs)
    recs = recs[: a.n_seq]
    t0 = time.time()

    def job(r):
        o, n = r["offset"], r["length"]
        res = score(a.server, a.model, toks[o:o + n], roles[o:o + n])
        res.update({k: r[k] for k in ("task_id", "pair", "attempt", "correct")})
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(job, recs))
    with open(a.out, "a") as f:
        for r in out:
            f.write(json.dumps({"tag": a.tag, **r}) + "\n")
    ng = sum(r["n_gen"] for r in out)
    print(json.dumps({"tag": a.tag, "seqs": len(out), "gen_tokens": ng,
                      "nll_gen": sum(r["nll_gen"] for r in out) / ng,
                      "top1_gen": sum(r["top1_gen"] for r in out) / ng,
                      "nll_all": sum(r["nll_all"] for r in out) / sum(r["n_all"] for r in out),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
