#!/usr/bin/env python3
"""E043: one comparison table for the no-tool candidates (accuracy, tokens, dev NLL, answer writing, throughput).

  lowbit_summary.py <evals_results_dir> <e043_results_dir> [--md out.md] [--json out.json]
Sources: evals/results/e043nt_<tag>.jsonl (ARC), results/*nll*dev*.jsonl (dev NLL summaries), *copy_probe*.jsonl
(answer writing at T=1), results/server_series.json (RTX-box server stats, segmented by the runtime mask in effect),
and E042's 4xL4 measurements (dummy weights, same shapes; copied from its messages, see L4 below).
Partial runs are compared with the full model on the same finished pairs (finished pairs skew easier).
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

FULL_NLL = None
# tag: (label, experts x format, avg expert bits, Engram, dev-NLL tag, probe tag, server run(s) + mask substring)
RUNS = [
    ("full_a1", "full model", "384 x MXFP4", 4.25, "full table", "full", "full", [("tp4A", "None")]),
    ("e72_w10_eng48", "E72 x10", "72 x MXFP4", 4.25, "48 GiB", "e72_w10", "e72_w10", [("tp4A", "w10_e72"), ("tp4B", "w10_e72")]),
    ("e64_w10_eng48", "E64 x10", "64 x MXFP4", 4.25, "48 GiB", "e64_w10", "e64_w10", [("tp4A", "w10_e64")]),
    ("e64_w20_eng48", "E64 x20", "64 x MXFP4", 4.25, "48 GiB", "e64_w20", "e64_w20", [("tp4B", "w20_e64")]),
    ("e128_u2_gptq_eng56", "E128 2-bit GPTQ", "128 x uint2 (w13 g128, w2 g64)", 2.19, "30% (56.6 GiB)", "e128_u2_gptq",
     "e128_u2_gptq", [("tp4B_e128_gptq", "w10_e128")]),
    ("e96_u2u3_gptq_eng56", "E96 2/3-bit GPTQ", "96 x uint2 / uint3 down", 2.53, "30% (56.6 GiB)", "e96_u2u3_gptq",
     "e96_u2u3_gptq", [("tp4A_e96_gptq", "w10_e96"), ("tp4B_e96_gptq_export", "None")]),
    ("e128_u2_gptq_fp4eng", "E128 2-bit GPTQ, FP4 Engram", "128 x uint2", 2.19, "30%, MXFP4", "e128_u2_gptq_fp4eng",
     "e128_u2_gptq_fp4eng", [("tp4A_e128_fp4eng", "None")]),
    ("e128_unsloth_iq2_eng56", "E128 unsloth UD-IQ2_XXS experts", "128 x IQ1_S-IQ3_XXS mix (as uint6)", 2.3,
     "30% (56.6 GiB)", "e128_unsloth_iq2", "e128_unsloth_iq2", [("tp4B_e128_unsloth", "None")]),
]
# E042 (2026-10-24), 4xL4, dummy weights at the same shapes, Engram 30%, util 0.93, largest C under an 80% KV guard, 0
# preemptions; attempts/h = output / decode rate + 9.4K prompt / ~1,270 tok/s prefill (crude, serial)
L4 = {
    "full_a1": "does not fit",
    "e72_w10_eng48": "16.56 GiB/GPU; util .93, own shapes: C52, 399.5 tok/s, ~42 attempts/h",
    "e128_u2_gptq_eng56": "16.87 GiB/GPU; util .93, own 331 shapes: C48, 419.2 tok/s, ~34 attempts/h",
    "e96_u2u3_gptq_eng56": "15.18 GiB/GPU; util .93, E72 shapes: C80, 558.0 tok/s, ~57-65 attempts/h at 27-31K out (~50/h at its own 37.4K)",
    "e128_u2_gptq_fp4eng": "as E128 (Engram is host RAM: 56.6 -> ~30 GiB)",
}


def ok(r):
    return r.get("correct") in (True, "True")


def tokens(r):
    u = r.get("usage") or {}
    if isinstance(u, str):
        u = json.loads(u.replace("'", '"').replace("None", "null"))
    return u.get("completion_tokens") or 0


def load_eval(path):
    rows = [json.loads(l) for l in open(path)] if path.exists() else []
    return [r for r in rows if "error" not in r]


def nll_tags(res):
    """Mean NLL per generated token per tag, from the per-sequence records of the 64-trace no-tool dev screen."""
    acc = collections.defaultdict(lambda: [0.0, 0])
    for name in ("notool_nll_nt2_dev.jsonl", "notool_nll_nt2_dev_B.jsonl", "lowbit_nll_dev.jsonl"):
        for l in open(res / name):
            r = json.loads(l)
            if "task_id" in r:
                acc[r["tag"]][0] += r["nll_gen"]
                acc[r["tag"]][1] += r["n_gen"]
    return {t: s / n for t, (s, n) in acc.items() if n}


def probe_tags(res):
    out = collections.defaultdict(lambda: [0, 0])
    for f in sorted(res.glob("*copy_probe*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            for s in r["samples"]:
                if s["temperature"] == 1.0:
                    out[r["config"]][0] += s["exact"]
                    out[r["config"]][1] += 1
    return out


def server_tps(series, runs):
    gen, run = [], []
    for name, mask in runs:
        s = series.get(name)
        if not s:
            continue
        loads = sorted(s["ctrl_loads"], key=lambda c: c["at_stats_index"])
        cur = "None"
        li = 0
        for i, row in enumerate(s["stats"]):
            while li < len(loads) and loads[li]["at_stats_index"] <= i:
                cur = loads[li]["experts"]
                li += 1
            if name == "tp4A" and mask == "None" and (row[0], row[1]) < ("10-23", "20:30:00"):
                continue  # tool-round full-model evals on the same server before the no-tool round
            if mask not in str(cur) or row[3] <= 0 or row[4] < 8:
                continue
            gen.append(row[3])
            run.append(row[4])
    return (statistics.mean(gen), statistics.mean(run), len(gen) * 10 / 60) if gen else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals")
    ap.add_argument("results")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev, res = Path(a.evals), Path(a.results)
    nll, probe = nll_tags(res), probe_tags(res)
    series = json.load(open(res / "server_series.json"))
    full = {(r["task_id"], str(r["pair"]), str(r["attempt"])): ok(r) for r in load_eval(ev / "e043nt_full_a1.jsonl")}
    base = nll.get("full")
    rows, js = [], {}
    for tag, label, fmt, bpw, eng, ntag, ptag, srv in RUNS:
        rr = load_eval(ev / f"e043nt_{tag}.jsonl")
        if not rr and tag not in L4:
            continue
        a0 = [r for r in rr if str(r["attempt"]) == "0"]
        byp = collections.defaultdict(dict)
        for r in rr:
            byp[(r["task_id"], str(r["pair"]))][str(r["attempt"])] = ok(r)
        two = [k for k, v in byp.items() if len(v) >= 2]
        tw1 = collections.defaultdict(list)
        for r in a0:
            tw1[r["task_id"]].append(ok(r))
        tw2 = collections.defaultdict(list)
        for k in two:
            tw2[k[0]].append(byp[k]["0"] or byp[k]["1"])
        same1 = [k for k in byp if "0" in byp[k]]
        tok = sorted(tokens(r) for r in rr)
        tps = server_tps(series, srv)
        d = {"label": label, "format": fmt, "expert_bits": bpw, "engram": eng, "attempts": len(rr),
             "pairs_a0": len(a0), "pass1": sum(map(ok, a0)) / len(a0) if a0 else None,
             "pass1_tw": sum(sum(v) / len(v) for v in tw1.values()) / len(tw1) if tw1 else None,
             "pairs_both": len(two),
             "pass2_tw": sum(sum(v) / len(v) for v in tw2.values()) / len(tw2) if tw2 else None,
             "same_pairs_pass1": [sum(byp[k]["0"] for k in same1), sum(full.get((*k, "0"), False) for k in same1), len(same1)],
             "same_pairs_pass2": [sum(byp[k]["0"] or byp[k]["1"] for k in two),
                                  sum(full.get((*k, "0"), False) or full.get((*k, "1"), False) for k in two), len(two)],
             "tokens_mean": statistics.mean(tok) if tok else None, "tokens_median": statistics.median(tok) if tok else None,
             "tokens_p90": tok[int(0.9 * (len(tok) - 1))] if tok else None,
             "length_cut": sum(r.get("finish_reason") == "length" for r in rr),
             "unparsed": sum(r.get("pred") in (None, "", "None") for r in rr),
             "seconds_mean": statistics.mean(float(r.get("seconds", 0)) for r in rr) if rr else None,
             "dev_nll": nll.get(ntag), "dev_nll_vs_full": (nll[ntag] / base - 1) if base and ntag in nll else None,
             "answer_probe": probe[ptag] if probe[ptag][1] else None,
             "rtx_gen_tps": tps[0] if tps else None, "rtx_running": tps[1] if tps else None,
             "l4": L4.get(tag, "")}
        js[tag] = d
        pct = lambda x: f"{100 * x:.1f}%" if x is not None else "-"
        rows.append([label, fmt, f"{bpw:.2f}", eng,
                     f"+{100 * d['dev_nll_vs_full']:.1f}%" if d["dev_nll_vs_full"] is not None else "-",
                     f"{d['answer_probe'][0]}/{d['answer_probe'][1]}" if d["answer_probe"] else "-",
                     f"{d['pairs_a0']}/167", pct(d["pass1"]), pct(d["pass1_tw"]),
                     f"{pct(d['pass2_tw'])} ({d['pairs_both']})" if d["pairs_both"] else "-",
                     f"{d['same_pairs_pass1'][0]} vs {d['same_pairs_pass1'][1]} /{d['same_pairs_pass1'][2]}" if tag != "full_a1" else "-",
                     f"{d['same_pairs_pass2'][0]} vs {d['same_pairs_pass2'][1]} /{d['same_pairs_pass2'][2]}" if tag != "full_a1" and two else "-",
                     f"{d['tokens_mean'] / 1000:.1f}K / {d['tokens_median'] / 1000:.1f}K / {d['tokens_p90'] / 1000:.1f}K" if tok else "-",
                     f"{d['length_cut']} / {d['unparsed']}",
                     f"{d['rtx_gen_tps']:,.0f} @{d['rtx_running']:.0f}" if tps else "-", d["l4"]])
    head = ["model", "experts x format", "expert bits", "Engram", "dev NLL vs full", "answer written (T=1)",
            "pairs done", "pass@1", "pass@1 task-w", "pass@2 task-w (pairs)", "same pairs pass@1: this vs full",
            "same pairs pass@2: this vs full", "out tokens mean / median / p90", "length cut / unparsed",
            "RTX gen tok/s @ running (4 GPUs)", "4xL4 (E042)"]
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n")
    if a.json:
        json.dump(js, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
