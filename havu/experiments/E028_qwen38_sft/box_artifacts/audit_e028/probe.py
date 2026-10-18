"""Probe masked-position divergence: top-5 distributions at the largest-|dlp| positions; entropy-binned |dlp|."""
import json
import numpy as np
import requests
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.8-27B-e028-sft")
ref = json.load(open("/workspace/e028/tinker_reference.json"))
SERVERS = {"bf16": "http://127.0.0.1:30031", "fp8block": "http://127.0.0.1:30021", "modelopt": "http://127.0.0.1:30002"}


def run(url, toks):
    r = requests.post(f"{url}/generate", json={"input_ids": toks, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                               "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 5}, timeout=1800)
    m = r.json()["meta_info"]
    lp = np.array([np.nan if x[0] is None else x[0] for x in m["input_token_logprobs"]])
    top = [t or [] for t in m["input_top_logprobs"]]
    return lp, top


for si in (0, 5):
    s = ref["sequences"][si]
    toks = s["tokens"]
    w = np.array(s["weights"]) > 0
    t = np.array([np.nan if x is None else x for x in s["lp_sft"]], dtype=float)
    out = {k: run(u, toks) for k, u in SERVERS.items()}
    lb, topb = out["bf16"]
    pmax = np.array([np.exp(tt[0][0]) if tt else np.nan for tt in topb])
    pos = np.arange(len(toks)) > 0
    print(f"seq {si} {s['tag']}")
    for k in ("fp8block", "modelopt"):
        d = np.abs(out[k][0] - lb)
        dt = np.abs(t - lb)
        for name, m in (("masked", ~w & pos), ("trained", w & pos)):
            # bin by actual-token logprob under bf16
            bins = [(-1e9, -10), (-10, -3), (-3, -0.1), (-0.1, 1)]
            row = []
            for a, b in bins:
                mm = m & (lb >= a) & (lb < b)
                row.append(f"lp[{a:g},{b:g}) n={mm.sum()} {k}|d|={np.nanmean(d[mm]) if mm.sum() else 0:.3f} tinker|d|={np.nanmean(dt[mm]) if mm.sum() else 0:.3f}")
            print(f"  {k} {name}: " + " ; ".join(row))
        # argmax agreement at masked positions
        tq = np.array([tt[0][1] if tt else -1 for tt in out[k][1]])
        tb = np.array([tt[0][1] if tt else -1 for tt in topb])
        m = ~w & pos
        print(f"  {k} masked top1 agree={np.mean(tq[m] == tb[m]):.4f}  trained top1 agree={np.mean(tq[w & pos] == tb[w & pos]):.4f}")
    d = np.abs(out["fp8block"][0] - lb)
    for i in np.argsort(-np.nan_to_num(d))[:4]:
        ctx = tok.decode(toks[max(0, i - 30):i]).replace("\n", "\\n")
        print(f"  pos {i} tok={tok.decode([toks[i]])!r} w={int(w[i])} ctx=...{ctx[-80:]!r}")
        for k in SERVERS:
            print(f"     {k:9s} lp={out[k][0][i]:.2f} top5={[(tok.decode([x[1]]), round(x[0], 2)) for x in out[k][1][i]]}")
        print(f"     tinker    lp={t[i]:.2f}")
