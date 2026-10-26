"""Final per-run table: age, max KV, thrash windows, gate holds, running min-max, records, records/h, accuracy vs E043, mem peak."""
import collections, json, os, re, statistics, sys
P = os.path.dirname(os.path.abspath(__file__)); A = "/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/pods/runs"
ROOT = "/home/vu/Workspace/ARC-AGI-2"
sol = json.load(open(f"{ROOT}/data/arc-agi_evaluation_solutions.json"))
ref = collections.defaultdict(list)
for r in map(json.loads, open(f"{ROOT}/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80.jsonl")):
    if "task_id" in r:
        ref[(r["task_id"], r["pair"])].append(bool(r.get("correct")))
pat = re.compile(r"\[(\S+) \+([\d.]+)h\] server: (-?[\d.]+) gen tok/s, prefix hits (\S+), uncached (-?[\d.]+) tok/s, KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?; (\d+) attempts done")
def first(*paths):
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
print("| run | KV pool | age h | max KV (at h) | thrash windows (first) | gate holds | running min-max | records | rec/h | pair acc vs E043 | None | mem GiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in sys.argv[1:]:
    hf = first(f"{P}/old/hist_{run}.txt", f"{P}/hist_{run}.txt")
    L = open(hf).read().splitlines() if hf else []
    W = []
    for m in map(pat.search, L):
        if m and m.group(4) != "None":
            g = m.groups(); W.append(dict(h=float(g[1]), tps=float(g[2]), hits=float(g[3]), unc=float(g[4]), kv=float(g[5]), rn=float(g[6])))
    done = [l for l in L if "done at" in l]
    age = float(re.search(r"done at \+([\d.]+)h", done[-1]).group(1)) if done else (W[-1]["h"] if W else 0)
    pool = next((re.search(r"GPU KV cache size: ([\d,]+)", l).group(1) for l in L if "GPU KV cache size" in l), "?")
    steady = [w["tps"] for w in W if w["h"] >= 0.3 and w["kv"] < 0.75 and w["tps"] > 0]
    plat = statistics.median(steady) if steady else None
    thr = [w for w in W if w["h"] >= 0.4 and ((plat and w["tps"] < 0.75 * plat and w["kv"] > 0.8) or (w["unc"] > 350 and w["hits"] < 0.9))]
    pool_thr = [w for w in thr if w["kv"] > 0.8]
    mk = max(W, key=lambda w: w["kv"]) if W else None
    rr = [w["rn"] for w in W if w["h"] >= 0.3]
    holds = sum("KV gate" in l for l in L)
    af = first(f"{A}/{run}/out/dsv41_logs/attempts.jsonl", f"{P}/acc/{run}.jsonl", f"{P}/t7/attempts.jsonl" if run == "c36_deadline" else "")
    recs = [r for r in map(json.loads, open(af)) if "task_id" in r] if af else []
    ok = tot = 0; exp = []; none = 0
    for r in recs:
        if r.get("pred") is None or r.get("error"):
            none += r.get("pred") is None; continue
        t = r["task_id"]
        items = list(enumerate(r["pred"])) if r.get("multi_test") and isinstance(r["pred"], list) else [(r.get("pair", 0), r["pred"])]
        for q, g in items:
            tot += 1; ok += g == sol[t][q]
            if ref.get((t, q)): exp.append(sum(ref[(t, q)]) / len(ref[(t, q)]))
    mf = first(f"{A}/{run}/mem.log", f"{P}/old/mem_{run}.log", f"{P}/mem_{run}.log")
    mem = [float(x) for x in re.findall(r"rss_gib=([\d.]+)", open(mf).read())][3:] if mf else []
    acc = f"{ok}/{tot} {100*ok/max(1,tot):.1f}% vs {100*sum(exp)/max(1,len(exp)):.1f}%" if tot else "-"
    th = f"{len(thr)} ({len(pool_thr)} at KV>0.8)" + (f", first {thr[0]['h']:.2f}h" if thr else "")
    print(f"| {run} | {pool} | {age:.2f} | {mk['kv'] if mk else '-'} ({mk['h'] if mk else '-'}) | {th} | {holds} | "
          f"{min(rr) if rr else 0:.0f}-{max(rr) if rr else 0:.0f} | {len(recs)} | {len(recs)/max(age,0.01):.1f} | {acc} | {none} | {max(mem) if mem else '-'} |")
