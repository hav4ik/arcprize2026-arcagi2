"""E044: no-tool rollouts vs the tool rollouts on the same pairs (E031 for arc2_train/nvarc_b0, E036+E038 for nvarc_b1).
Usage (repo root): .venv/bin/python experiments/E044_notool_rollouts/scripts/analyze.py [arc2_train nvarc_b0 nvarc_b1]"""
import gzip, json, re, statistics as st, sys, zlib
R = "evals/results/"
P = R + "tools_persistent_grid_lean2sub_deepseek-flash_low_"
TOOL = {"arc2_train": [P + "e031_arc2_train.jsonl"], "nvarc_b0": [P + "e031_nvarc_b0.jsonl"],
        "nvarc_b1": [P + "e036_nvarc_b1.jsonl", P + "e036_nvarc_b1x.jsonl", R + "e038_deepinfra_nvarc_b1r.jsonl"]}
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

def load(files):
    """Later files override earlier ones; rows with an API error are dropped."""
    d = {}
    for f in files:
        try:
            for r in map(json.loads, open(f)):
                d[(r["task_id"], r["pair"])] = r
        except FileNotFoundError:
            pass
    return {k: r for k, r in d.items() if "error" not in r}

def q(xs, p): return sorted(xs)[int(p * (len(xs) - 1))]
# Same provider priority as hf_upload.py (the first provider wins); arc2_train on DeepSeek is the separate sample2.
PROV = {"arc2_train": ["deepinfra"], "nvarc_b0": ["deepinfra", "deepseek"],
        "nvarc_b1": ["deepinfra", "inferencenet", "deepseek"], "arc2_train_sample2": ["deepseek"]}
for s in sys.argv[1:] or ["arc2_train", "nvarc_b0", "nvarc_b1", "arc2_train_sample2"]:
    base = s.replace("_sample2", "")
    per = {p: load([R + f"e044_notool_{p}_{base}.jsonl"]) for p in PROV[s]}
    nt = {}
    for p in reversed(PROV[s]): nt.update(per[p])
    if not nt:
        continue
    s = base if s != "arc2_train_sample2" else s
    tl = load(TOOL[base])
    n = len(nt); c = sum(bool(r["correct"]) for r in nt.values())
    out = [r["usage"]["completion_tokens"] for r in nt.values()]
    cost = sum(r.get("est_cost_usd") or 0 for r in nt.values())
    fr = {}
    for r in nt.values(): fr[r["finish_reason"]] = fr.get(r["finish_reason"], 0) + 1
    zr, cjk, empty = [], 0, 0
    for r in nt.values():
        t = gzip.open(r["reasoning_path"], "rt").read()
        empty += not t.strip(); cjk += len(CJK.findall(t)) > 50
        if len(t) > 2000: zr.append(len(zlib.compress(t.encode())) / len(t.encode()))
    pk = [k for k in nt if k in tl]
    both = sum(bool(nt[k]["correct"]) and bool(tl[k]["correct"]) for k in pk)
    only_nt = sum(bool(nt[k]["correct"]) and not tl[k]["correct"] for k in pk)
    only_tl = sum(not nt[k]["correct"] and bool(tl[k]["correct"]) for k in pk)
    tl_c = sum(bool(tl[k]["correct"]) for k in pk)
    tl_out = [tl[k]["usage"]["completion"] for k in pk]
    print(f"== {s}: {n} no-tool rollouts, correct {c} ({100*c/n:.1f}%), unparsed {sum(r['pred'] is None for r in nt.values())}, finish {fr}")
    print(f"   out tok mean {st.mean(out)/1e3:.1f}k median {st.median(out)/1e3:.1f}k p90 {q(out,.9)/1e3:.1f}k max {max(out)/1e3:.1f}k"
          f" | ${cost:.2f} (${1000*cost/n:.1f}m/attempt)")
    for p in PROV[s]:
        used = [k for k in nt if nt[k] is per[p].get(k)]
        if used:
            pk_p = [k for k in used if k in tl]
            print(f"   {p}: {len(used)} rows, correct {100*sum(bool(nt[k]['correct']) for k in used)/len(used):.1f}%"
                  f" (tool on same pairs {100*sum(bool(tl[k]['correct']) for k in pk_p)/max(1,len(pk_p)):.1f}%)"
                  f", ${sum(nt[k].get('est_cost_usd') or 0 for k in used):.2f}")
    print(f"   text: empty thinking {empty}, CJK>50 {cjk}, zlib median {st.median(zr):.3f} min {min(zr):.3f} (<0.10: {sum(z<0.10 for z in zr)})")
    if pk:
        print(f"   vs tool rollouts on the same {len(pk)} pairs: tool correct {100*tl_c/len(pk):.1f}% (out tok median {st.median(tl_out)/1e3:.1f}k)"
              f" | no-tool {100*sum(bool(nt[k]['correct']) for k in pk)/len(pk):.1f}% | both {both}, only no-tool {only_nt}, only tool {only_tl}"
              f", neither {len(pk)-both-only_nt-only_tl}")
