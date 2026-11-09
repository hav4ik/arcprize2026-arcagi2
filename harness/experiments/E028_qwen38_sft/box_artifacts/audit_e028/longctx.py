"""Long-context logprob parity: FP8block and ModelOpt FP8 vs BF16 (all SGLang) on long E028 training sequences."""
import json
import sys
import time
import numpy as np
import requests

SERVERS = {"bf16": "http://127.0.0.1:30031", "fp8block": "http://127.0.0.1:30021", "modelopt": "http://127.0.0.1:30002"}
d = np.load("/workspace/audit_e028/longseq.npz")
BINS = [0, 8192, 16384, 32768, 49152, 65536]


def run(url, toks):
    r = requests.post(f"{url}/generate", json={
        "input_ids": toks, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 1}, timeout=3600)
    r.raise_for_status()
    m = r.json()["meta_info"]
    lps = m["input_token_logprobs"]
    tops = m["input_top_logprobs"]
    assert len(lps) == len(toks)
    lp = np.array([np.nan if x[0] is None else x[0] for x in lps])
    top = np.array([-1 if not t else t[0][1] for t in tops])
    return lp, top


res = {}
for k in range(4):
    toks = d[f"t{k}"].tolist()
    w = d[f"w{k}"].astype(bool)
    out = {}
    for name, url in SERVERS.items():
        t0 = time.time()
        out[name] = run(url, toks)
        print(f"seq{k} len={len(toks)} {name} {time.time()-t0:.0f}s", flush=True)
    lb, tb = out["bf16"]
    pos = np.arange(len(toks))
    for name in ("fp8block", "modelopt"):
        lq, tq = out[name]
        rows = []
        for a, b in zip(BINS[:-1], BINS[1:]):
            m = w & (pos >= a) & (pos < b) & (pos > 0)
            if m.sum() < 50:
                continue
            nb, nq = -lb[m].mean(), -lq[m].mean()
            dl = np.abs(lq[m] - lb[m])
            mall = (pos >= max(a, 1)) & (pos < b)
            agree = (tq[mall] == tb[mall]).mean()
            rows.append((f"{a//1024}-{b//1024}k", int(m.sum()), round(nb, 4), round(nq - nb, 4), round(dl.mean(), 4),
                         round(float(np.percentile(dl, 99.9)), 3), round(float(agree), 4)))
        mt = w & (pos > 0)
        print(f"  {name}: trained NLL bf16={-lb[mt].mean():.4f} {name}={-lq[mt].mean():.4f} delta={(-lq[mt].mean())-(-lb[mt].mean()):+.4f} "
              f"top1 agree(all pos)={(tq[1:]==tb[1:]).mean():.4f}")
        for r in rows:
            print("    bin %-9s n=%6d nll_bf16=%.4f dNLL=%+.4f mean|dlp|=%.4f p99.9|dlp|=%.3f top1agree=%.4f" % r)
        res[f"seq{k}_{name}"] = rows
json.dump(res, open("/workspace/audit_e028/longctx_result.json", "w"), indent=1)
