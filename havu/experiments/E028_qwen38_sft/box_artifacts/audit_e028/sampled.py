"""Sampled-generation loop rate and quality: 4 ARC prompts x N samples, BF16 vs FP8block vs ModelOpt."""
import json
import sys
import concurrent.futures as cf
import numpy as np
import requests

SERVERS = {"bf16": "http://127.0.0.1:30031", "fp8block": "http://127.0.0.1:30021", "modelopt": "http://127.0.0.1:30002"}
ref = json.load(open("/workspace/e028/tinker_reference.json"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXN = 4000
SP = {"max_new_tokens": MAXN, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stop_token_ids": [248046]}


def gen(url, prompt, seed):
    r = requests.post(f"{url}/generate", json={"input_ids": prompt, "sampling_params": dict(SP, sampling_seed=seed)}, timeout=7200)
    r.raise_for_status()
    o = r.json()
    return list(o.get("output_ids") or o["meta_info"]["output_ids"])


def score(url, prompt, out):
    r = requests.post(f"{url}/generate", json={"input_ids": prompt + out, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                               "return_logprob": True, "logprob_start_len": len(prompt)}, timeout=3600)
    lps = [x[0] for x in r.json()["meta_info"]["input_token_logprobs"] if x[0] is not None]
    return float(-np.mean(lps[-len(out):]))


def rep_frac(ids, n=32):
    s, rep = set(), 0
    grams = [tuple(ids[i:i + n]) for i in range(0, max(0, len(ids) - n))]
    for i, g in enumerate(grams):
        if g in s and i > len(grams) // 2:
            rep += 1
        s.add(g)
    return rep / max(1, len(grams) // 2)


res = {k: [] for k in SERVERS}
jobs = [(k, pi, j) for k in SERVERS for pi in range(len(ref["prompts"])) for j in range(N)]
with cf.ThreadPoolExecutor(64) as ex:
    futs = {ex.submit(gen, SERVERS[k], ref["prompts"][pi]["prompt"], 1000 * pi + j): (k, pi, j) for k, pi, j in jobs}
    outs = {futs[f]: f.result() for f in cf.as_completed(futs)}
with cf.ThreadPoolExecutor(16) as ex:
    nll = dict(zip(outs, ex.map(lambda kv: score(SERVERS["bf16"], ref["prompts"][kv[0][1]]["prompt"], kv[1]), outs.items())))
for (k, pi, j), o in outs.items():
    res[k].append({"p": pi, "len": len(o), "end": bool(o and o[-1] == 248046), "rep": rep_frac(o), "nll_bf16": nll[(k, pi, j)]})
for k, v in res.items():
    rep = np.array([x["rep"] for x in v])
    print(f"{k:9s} n={len(v)} loops(rep32>0.3)={int((rep > 0.3).sum())} mean rep32={rep.mean():.3f} ended(<4000, im_end)={sum(x['end'] for x in v)} "
          f"mean len={np.mean([x['len'] for x in v]):.0f} mean NLL under bf16={np.mean([x['nll_bf16'] for x in v]):.4f}")
json.dump(res, open("/workspace/audit_e028/sampled_result.json", "w"), indent=1)
