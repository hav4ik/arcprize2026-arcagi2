"""E028 parity check: served models (SGLang) vs Tinker's reference (tinker_reference.json from tinker_reference.py).

For each served model, against the matching Tinker reference ("sft" = adapter applied on Tinker, "base"):
- logprob parity on fixed training sequences: NLL over trained positions (and its difference to Tinker's), mean and
  max |Δ logprob| over all positions, and the fraction of positions within 0.05;
- greedy parity on first-turn prompts (temperature 0, same max tokens): number of leading tokens identical to
  Tinker's greedy output.

  python parity_check.py tinker_reference.json sft=http://127.0.0.1:30002:sft base=http://127.0.0.1:30001:base
  (each arg: label=url:reference, reference in {sft, base})
"""
import json
import sys
import time

import numpy as np
import requests


def served_logprobs(url, tokens):
    r = requests.post(f"{url}/generate", json={
        "input_ids": tokens, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "logprob_start_len": 0}, timeout=1800)
    r.raise_for_status()
    lps = r.json()["meta_info"]["input_token_logprobs"]  # [logprob, token_id, text]; position 0 has None
    assert len(lps) == len(tokens), (len(lps), len(tokens))
    assert all(tid == t for (_, tid, *_), t in zip(lps, tokens))
    return [lp for lp, *_ in lps]


def served_greedy(url, prompt, max_new, stop_ids):
    r = requests.post(f"{url}/generate", json={
        "input_ids": prompt, "sampling_params": {"max_new_tokens": max_new, "temperature": 0,
                                                 "stop_token_ids": stop_ids, "skip_special_tokens": False},
        "return_logprob": False}, timeout=3600)
    r.raise_for_status()
    out = r.json()
    ids = out.get("output_ids") or out["meta_info"].get("output_ids")
    return list(ids)


def first_diff(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ref = json.load(open(sys.argv[1]))
    servers = {}
    for a in sys.argv[2:]:
        label, rest = a.split("=", 1)
        url, which = rest.rsplit(":", 1)
        servers[label] = (url, which)
    report = {"sequences": [], "greedy": []}
    print("%-30s %-7s %8s %8s %8s %9s %9s %7s" % ("sequence", "model", "nll_ref", "nll", "Δnll", "mean|Δlp|",
                                                  "max|Δlp|", "≤0.05"))
    for s in ref["sequences"]:
        tr = [j for j, w in enumerate(s["weights"]) if w]
        for label, (url, which) in servers.items():
            t0 = time.time()
            lp = served_logprobs(url, s["tokens"])
            a = np.array(lp[1:], dtype=np.float64)
            b = np.array(s[f"lp_{which}"][1:], dtype=np.float64)
            d = np.abs(a - b)
            nll = float(-np.mean([lp[j] for j in tr]))
            row = {"tag": s["tag"], "idx": s["idx"], "model": label, "ref": which, "nll_ref": s[f"nll_{which}"],
                   "nll": nll, "d_nll": nll - s[f"nll_{which}"], "mean_abs_dlp": float(d.mean()),
                   "max_abs_dlp": float(d.max()), "frac_within_0.05": float((d <= 0.05).mean()),
                   "seconds": round(time.time() - t0, 1)}
            report["sequences"].append(row)
            print("%-30s %-7s %8.4f %8.4f %+8.4f %9.4f %9.3f %6.1f%%" % (
                s["tag"], label, row["nll_ref"], nll, row["d_nll"], row["mean_abs_dlp"], row["max_abs_dlp"],
                100 * row["frac_within_0.05"]), flush=True)
    print()
    for p in ref["prompts"]:
        for label, (url, which) in servers.items():
            ref_ids = p[f"greedy_{which}"]
            ids = served_greedy(url, p["prompt"], len(ref_ids), p["stop"])
            k = first_diff(ids, ref_ids)
            report["greedy"].append({"tag": p["tag"], "task_id": p["task_id"], "model": label, "ref": which,
                                     "ref_len": len(ref_ids), "len": len(ids), "identical_prefix": k})
            print(f"{p['tag']:<20} {p['task_id']:<20} {label:<7} identical to Tinker {which} greedy for "
                  f"{k}/{len(ref_ids)} tokens", flush=True)
    json.dump(report, open(sys.argv[1].replace(".json", "") + "_parity_" + "_".join(servers) + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
