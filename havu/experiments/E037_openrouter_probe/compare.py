"""E037: OpenRouter-provider rollouts vs DeepSeek-native E036 rollouts on the same pairs (same harness)."""
import gzip, json, statistics as st, sys
R = "evals/results/"
ds = {(r["task_id"], r["pair"]): r for r in map(json.loads, open(R + "tools_persistent_grid_lean2sub_deepseek-flash_low_e036_nvarc_b1.jsonl"))}
orr = [r for r in map(json.loads, open(R + sys.argv[1])) if "error" not in r]

def turn_stats(r):
    t = json.load(gzip.open(r["trace_path"], "rt"))
    ev = [e for e in t["events"] if "usage" in e]
    reas = [((e["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")) for e in ev]
    rchars = sum(len(m.get("reasoning_content", "")) for m in t["messages"] if m.get("role") == "assistant")
    return max(e["usage"]["completion_tokens"] for e in ev), sum(x or 0 for x in reas), rchars, sum(e["finish_reason"] == "length" for e in ev)

def q(xs, p): return st.quantiles(xs, n=20)[p] if len(xs) > 2 else max(xs)
rows = []
for r in orr:
    d = ds[(r["task_id"], r["pair"])]
    rows.append((r, d, turn_stats(r), turn_stats(d)))
for name, i in (("OpenRouter", 0), ("DeepSeek", 1)):
    rs = [x[i] for x in rows]; ts = [x[2 + i] for x in rows]
    out = [r["usage"]["completion"] for r in rs]; secs = [r["seconds"] for r in rs]
    cost = sum(r["est_cost_usd"] for r in rs)
    print(f"{name:10s} n={len(rs)} correct {sum(bool(r['correct']) for r in rs)} train_pass {sum(r.get('train_pass') is True for r in rs)}"
          f" | out tok/attempt mean {st.mean(out)/1e3:.1f}k median {st.median(out)/1e3:.1f}k p95 {q(out,18)/1e3:.1f}k max {max(out)/1e3:.1f}k"
          f" | reasoning chars/attempt median {st.median(t[2] for t in ts)/1e3:.0f}k | max single turn {max(t[0] for t in ts)/1e3:.1f}k"
          f" | length-stopped turns {sum(t[3] for t in ts)} | stops {dict((s, sum(r['stop_reason']==s for r in rs)) for s in {r['stop_reason'] for r in rs})}"
          f" | wall s median {st.median(secs):.0f} p95 {q(secs,18):.0f} | cost ${cost:.2f} (${1000*cost/len(rs):.1f}m/attempt)")
both = sum(bool(a["correct"]) and bool(b["correct"]) for a, b, *_ in rows)
print(f"paired: both {both}, only OR {sum(bool(a['correct']) and not b['correct'] for a,b,*_ in rows)}, only DS {sum(not a['correct'] and bool(b['correct']) for a,b,*_ in rows)}")
reas_tok = [x[2][1] for x in rows]
print(f"OpenRouter reported reasoning_tokens/attempt: median {st.median(reas_tok)/1e3:.1f}k, mean {st.mean(reas_tok)/1e3:.1f}k")
