#!/usr/bin/env python3
"""E066: teacher-forced NLL of held-out image rollouts (dev screen for keep sets), E043 nll_eval.py over chat.

Each dev rollout goes to /v1/chat/completions (mm_common.chat_body) with prompt_logprobs=0, max_tokens=1 and a
unique cache_salt, so the images are processed and no KV cached under another mask is reused. The returned prompt
logprobs must carry the offline stream's token at every position (a stream check). Reports NLL and top-1 agreement
on model-generated targets, all targets, and tool-call-block targets (</think> through EOS), per rollout, in the
nll_eval.py format so candidates compare paired against the full model.

  nll_eval_mm.py --tok DIR [...] --parquet e065_image.parquet --n-seq 64 --per-task --tag full --out dev_nll.jsonl
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from mm_common import KEY, chat_body, load_index, load_rows

THINK_END, EOS = 128822, 1


def score(server, body, ids, roles):
    r = requests.post(f"{server}/v1/chat/completions", json=body, timeout=7200)
    r.raise_for_status()
    plp = r.json()["prompt_logprobs"]
    if len(plp) != len(ids):
        raise SystemExit(f"server prompt has {len(plp)} positions, offline stream {len(ids)}")
    lp = np.zeros(len(ids))
    rank = np.zeros(len(ids), dtype=np.int64)
    for t in range(1, len(ids)):
        d = plp[t]
        e = d.get(str(int(ids[t]))) or d.get(int(ids[t]))
        if e is None:
            raise SystemExit(f"position {t}: offline token {int(ids[t])} not in the server's prompt logprobs")
        lp[t] = e["logprob"]
        rank[t] = e.get("rank") or 0
    g = roles[1:] == 1
    nll = -lp[1:]
    blk = np.zeros(len(ids), bool)
    on = False
    for t in range(1, len(ids)):
        if roles[t] != 1:
            on = False
        elif on or ids[t] == THINK_END:
            blk[t] = True
            on = ids[t] != EOS
    b = blk[1:]
    return {"n_gen": int(g.sum()), "nll_gen": float(nll[g].sum()), "top1_gen": int((rank[1:][g] == 1).sum()),
            "n_all": len(ids) - 1, "nll_all": float(nll.sum()),
            "n_blk": int(b.sum()), "nll_blk": float(nll[b].sum()), "top1_blk": int((rank[1:][b] == 1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--per-task", action="store_true", help="one random rollout per task family, --n-seq of them")
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=60000)
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--effort", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load_index(a.tok, a.max_len)
    recs.sort(key=lambda r: (r["task_id"], r["pair"], r["attempt"], r.get("batch") or "", r["dir"]))
    rng = random.Random(a.seed)
    if a.per_task:
        by_fam = {}
        for r in recs:
            by_fam.setdefault(r["parent"], []).append(r)
        fams = sorted(by_fam)
        rng.shuffle(fams)
        recs = [rng.choice(by_fam[f]) for f in fams[: a.n_seq]]
    else:
        rng.shuffle(recs)
        recs = recs[: a.n_seq]
    rows = load_rows(a.parquet)
    data = {d: (np.fromfile(f"{d}/tokens.u32", dtype=np.uint32), np.fromfile(f"{d}/roles.u8", dtype=np.uint8))
            for d in {r["dir"] for r in recs}}
    t0 = time.time()

    def job(r):
        o, n = r["offset"], r["length"]
        toks, roles = data[r["dir"]]
        body = chat_body(rows[tuple(r[k] for k in KEY)], a.model, a.effort, prompt_logprobs=0)
        res = score(a.server, body, toks[o:o + n], roles[o:o + n])
        res.update({k: r.get(k) for k in ("task_id", "pair", "attempt", "correct", "batch", "n_img")})
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
                      "nll_blk": sum(r["nll_blk"] for r in out) / max(1, sum(r["n_blk"] for r in out)),
                      "top1_blk": sum(r["top1_blk"] for r in out) / max(1, sum(r["n_blk"] for r in out)),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
