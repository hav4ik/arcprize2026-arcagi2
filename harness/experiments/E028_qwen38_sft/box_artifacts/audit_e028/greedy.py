"""Greedy generation sanity: BF16 vs FP8block vs ModelOpt on reference ARC prompts, plus cross-scoring under BF16."""
import json
import concurrent.futures as cf
import numpy as np
import requests
from transformers import AutoTokenizer

SERVERS = {"bf16": "http://127.0.0.1:30031", "fp8block": "http://127.0.0.1:30021", "modelopt": "http://127.0.0.1:30002"}
ref = json.load(open("/workspace/e028/tinker_reference.json"))
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.8-27B-e028-sft")
MAXN = 3000


def gen(url, prompt, stop):
    r = requests.post(f"{url}/generate", json={"input_ids": prompt, "sampling_params": {
        "max_new_tokens": MAXN, "temperature": 0, "stop_token_ids": stop, "skip_special_tokens": False}}, timeout=3600)
    r.raise_for_status()
    o = r.json()
    return list(o.get("output_ids") or o["meta_info"]["output_ids"])


def score(url, prompt, out):
    r = requests.post(f"{url}/generate", json={"input_ids": prompt + out, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                               "return_logprob": True, "logprob_start_len": len(prompt)}, timeout=3600)
    r.raise_for_status()
    lps = [x[0] for x in r.json()["meta_info"]["input_token_logprobs"] if x[0] is not None]
    return float(-np.mean(lps[-len(out):]))


def rep_frac(ids, n=32):
    # fraction of n-grams in the last half that already appeared earlier: crude loop detector
    s = set()
    rep = 0
    grams = [tuple(ids[i:i + n]) for i in range(0, max(0, len(ids) - n))]
    for i, g in enumerate(grams):
        if g in s and i > len(grams) // 2:
            rep += 1
        s.add(g)
    return rep / max(1, len(grams) // 2)


report = []
for p in ref["prompts"]:
    outs = {}
    with cf.ThreadPoolExecutor(3) as ex:
        fut = {k: ex.submit(gen, u, p["prompt"], p["stop"]) for k, u in SERVERS.items()}
        outs = {k: f.result() for k, f in fut.items()}
    row = {"task": p["task_id"], "prompt_len": len(p["prompt"])}
    for k, o in outs.items():
        k0 = next((i for i, (a, b) in enumerate(zip(o, outs["bf16"])) if a != b), min(len(o), len(outs["bf16"])))
        row[k] = {"len": len(o), "ends_im_end": bool(o and o[-1] == 248046), "same_prefix_as_bf16": k0,
                  "nll_under_bf16": round(score(SERVERS["bf16"], p["prompt"], o), 4),
                  "nll_under_self": round(score(SERVERS[k], p["prompt"], o), 4), "rep32": round(rep_frac(o), 3),
                  "has_tool_call": "<tool_call>" in tok.decode(o)}
        open(f"/workspace/audit_e028/greedy_{p['task_id']}_{k}.txt", "w").write(tok.decode(o))
    print(json.dumps(row), flush=True)
    report.append(row)
json.dump(report, open("/workspace/audit_e028/greedy_result.json", "w"), indent=1)
