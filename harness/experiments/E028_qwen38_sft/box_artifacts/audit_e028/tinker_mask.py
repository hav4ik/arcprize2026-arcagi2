"""Where do served-vs-Tinker logprob differences live? Split |dlp| by trained vs masked positions."""
import json
import numpy as np
import requests

ref = json.load(open("/workspace/e028/tinker_reference.json"))
SERVERS = {"bf16": "http://127.0.0.1:30031", "fp8block": "http://127.0.0.1:30021"}


def lps(url, toks):
    r = requests.post(f"{url}/generate", json={"input_ids": toks, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                               "return_logprob": True, "logprob_start_len": 0}, timeout=1800)
    return np.array([np.nan if x[0] is None else x[0] for x in r.json()["meta_info"]["input_token_logprobs"]])


for s in ref["sequences"][:8]:
    toks = s["tokens"]
    w = np.array(s["weights"]) > 0
    t = np.array([np.nan if x is None else x for x in s["lp_sft"]], dtype=float)
    out = {k: lps(u, toks) for k, u in SERVERS.items()}
    pos = np.arange(len(toks)) > 0
    line = f"{s['tag']:28s} n={len(toks)} trained={w.sum()} "
    for k, v in out.items():
        d = np.abs(v - t)
        big = np.where(pos & (d > 5))[0]
        line += f"| {k}: trained mean|d|={np.nanmean(d[w & pos]):.4f} masked mean|d|={np.nanmean(d[~w & pos]):.4f} n(|d|>5)={len(big)} "
    dd = np.abs(out["fp8block"] - out["bf16"])
    line += f"| fp8block-vs-bf16 trained={np.nanmean(dd[w & pos]):.4f} masked={np.nanmean(dd[~w & pos]):.4f}"
    print(line, flush=True)
    if s is ref["sequences"][0]:
        d = np.abs(out["bf16"] - t)
        idx = np.argsort(-np.nan_to_num(d))[:8]
        print("   top diffs (pos, token, w, tinker, bf16):", [(int(i), toks[i], int(w[i]), round(t[i], 2), round(out['bf16'][i], 2)) for i in idx])
        # contiguous structure of the masked-position differences
        m = ~w & pos
        print("   masked |d| by position quintile:", [round(float(np.nanmean(d[m & (np.arange(len(toks)) // (len(toks) // 5) == q)])), 3) for q in range(5)])
