"""E038 quality check: DeepInfra rollouts vs the DeepSeek-native E036 rollouts (same harness, same nvarc_b1 pool).
Usage: qc.py [results.jsonl]  (default evals/results/e038_deepinfra_nvarc_b1r.jsonl). Run from the repo root.
Difficulty control: new tasks are different tasks, so accuracy is compared with a parent-matched expectation
(each new task's parent's E036 correct rate). Text checks look for missing thinking, loops and language drift."""
import gzip, json, random, re, statistics as st, sys, zlib
from collections import defaultdict
R = "evals/results/"
MAN = "experiments/E025_train_rollouts/data/manifest.jsonl"
parent = {m["task_id"]: m["parent"] for m in map(json.loads, open(MAN)) if m["batch"] == "nvarc_b1"}
base = [r for f in ("b1", "b1x") for r in map(json.loads, open(f"{R}tools_persistent_grid_lean2sub_deepseek-flash_low_e036_nvarc_{f}.jsonl")) if "error" not in r]
new_all = list({(r["task_id"], r["pair"], r["attempt"]): r for r in map(json.loads, open(sys.argv[1] if len(sys.argv) > 1 else R + "e038_deepinfra_nvarc_b1r.jsonl"))}.values())  # resumed: last record wins
new = [r for r in new_all if "error" not in r]
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

def text_stats(r):
    t = json.load(gzip.open(r["trace_path"], "rt"))
    asst = [m for m in t["messages"] if m.get("role") == "assistant"]
    reas = [m.get("reasoning_content") or "" for m in asst]
    allr = "\n".join(reas)
    b = allr.encode()
    ev = [e for e in t["events"] if "usage" in e]
    return {"turns": len(asst), "empty_reas_turns": sum(not x.strip() for x in reas), "reas_chars": len(allr),
            "zratio": len(zlib.compress(b)) / max(1, len(b)) if len(b) > 2000 else None,
            "cjk": len(CJK.findall(allr)), "len_stops": sum(e.get("finish_reason") == "length" for e in ev)}

def q(xs, p): return st.quantiles(xs, n=20)[p] if len(xs) > 2 else max(xs)
def pct(a, n): return f"{100*a/max(1,n):.1f}%"

def summary(name, rs, ts):
    n = len(rs); out = [r["usage"]["completion"] for r in rs]
    c = sum(bool(r["correct"]) for r in rs); tp = sum(r.get("train_pass") is True for r in rs)
    ctp = sum(bool(r["correct"]) and r.get("train_pass") is True for r in rs)
    fp = sum(r.get("train_pass") is True and not r["correct"] for r in rs)
    zr = [t["zratio"] for t in ts if t["zratio"] is not None]
    stops = defaultdict(int)
    for r in rs: stops[r["stop_reason"]] += 1
    print(f"\n== {name}: n={n}")
    print(f"  correct {pct(c,n)} | train_pass {pct(tp,n)} | correct&train_pass {pct(ctp,n)} | train_pass-but-wrong {pct(fp,tp)} of train_pass")
    print(f"  out tok mean {st.mean(out)/1e3:.1f}k median {st.median(out)/1e3:.1f}k p95 {q(out,18)/1e3:.1f}k max {max(out)/1e3:.1f}k"
          f" | tool errors/call {sum(r['tool_errors'] for r in rs)/max(1,sum(r['tool_calls'] for r in rs)):.3f}")
    print(f"  stops {dict(stops)} | usage_incomplete {sum(not r.get('usage_complete', True) for r in rs)}")
    print(f"  [traces n={len(ts)}] turns w/o thinking {pct(sum(t['empty_reas_turns'] for t in ts), sum(t['turns'] for t in ts))}"
          f" | attempts w/o any thinking {sum(t['reas_chars'] == 0 for t in ts)} | length-stopped turns {sum(t['len_stops'] for t in ts)}"
          f" | thinking chars median {st.median(t['reas_chars'] for t in ts)/1e3:.1f}k")
    print(f"  zlib ratio (low = repetitive) median {st.median(zr):.3f} p5 {q(zr,0):.3f} min {min(zr):.3f} | attempts <0.10: {sum(z < 0.10 for z in zr)}"
          f" | attempts with CJK chars {sum(t['cjk'] > 0 for t in ts)} (>50: {sum(t['cjk'] > 50 for t in ts)})")
    return zr

random.seed(0)
bs = random.sample(base, min(800, len(base)))
print(f"new file rows {len(new_all)} (errors {len(new_all) - len(new)}: {[r.get('error','')[:80] for r in new_all if 'error' in r][:3]})")
summary("DeepSeek E036 (sample)", bs, [text_stats(r) for r in bs])
nts = [text_stats(r) for r in new]
summary("DeepInfra (new)", new, nts)
# parent-matched expected accuracy from E036
pc = defaultdict(list)
for r in base: pc[parent[r["task_id"]]].append(bool(r["correct"]))
m = [(bool(r["correct"]), st.mean(pc[parent[r["task_id"]]])) for r in new if pc.get(parent.get(r["task_id"]))]
obs, exp = sum(a for a, _ in m), sum(e for _, e in m)
sd = sum(e * (1 - e) for _, e in m) ** 0.5
print(f"\nparent-matched: observed correct {obs}/{len(m)} = {100*obs/len(m):.1f}% vs expected from E036 parents {100*exp/len(m):.1f}%"
      f"  (z = {(obs-exp)/sd:+.2f})")
flag = [r["task_id"] for r, t in zip(new, nts) if (t["zratio"] or 1) < 0.10 or t["cjk"] > 50 or t["reas_chars"] == 0]
print(f"flagged attempts (loops/CJK/no thinking): {len(flag)} {flag[:10]}")
