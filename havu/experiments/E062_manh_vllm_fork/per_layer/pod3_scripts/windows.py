"""Per-5-min windows of an external-server load run: server counters (scrape_metrics.sh, every 60 s) for gen/s,
prompt/s and the prefix split (GPU-hit / CPU-restored / recompute = queries - hits - external hits), the solver's
metrics.jsonl (KV, running/waiting, preemptions), KV-gate holds from solver.log, first attempts done/correct.
Usage: windows.py <run dir copy (out/dsv41_logs/...)> <metrics_scrape.txt> <solutions.json> [bucket_s=300]"""
import json, re, sys
from datetime import datetime, timezone

run, scrape, sols = sys.argv[1:4]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 300
L = f"{run}/out/dsv41_logs"
sol = json.load(open(sols))

# solver T0 (epoch) from the first stamped solver.log line "[HH:MM:SS +X.XXh]"
t0, holds = None, []
day = None
for line in open(f"{L}/solver.log", errors="replace"):
    m = re.match(r"\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\]", line)
    if not m:
        continue
    hh, mm, ss, eh = int(m[1]), int(m[2]), int(m[3]), float(m[4])
    if t0 is None:
        now = datetime.now(timezone.utc)
        base = now.replace(hour=hh, minute=mm, second=ss, microsecond=0).timestamp()
        if base > now.timestamp() + 60:
            base -= 86400
        t0 = base - eh * 3600
    if "KV gate:" in line and "holding" in line:
        holds.append(eh * 3600)

# scrape samples
samples, cur = [], None
for line in open(scrape):
    line = line.strip()
    if line.startswith("T "):
        cur = {"t": int(line.split()[1])}
        samples.append(cur)
    elif cur is not None and line:
        k, v = line.rsplit(" ", 1)
        try:
            cur[k.replace("vllm:", "")] = cur.get(k.replace("vllm:", ""), 0.0) + float(v)
        except ValueError:
            pass

atts = [json.loads(l) for l in open(f"{L}/attempts.jsonl")] if __import__("os").path.exists(f"{L}/attempts.jsonl") else []


def correct(r):
    t, p = r["task_id"], r.get("pair", 0)
    s = sol.get(t)
    if s is None:
        return None
    pr = r.get("pred")
    if len(s) > 1 and r.get("multi_test"):
        return isinstance(pr, list) and len(pr) > p and pr[p] == s[p]
    return pr == s[p] if p < len(s) else None


first = [r for r in atts if r.get("attempt") == 0 and r.get("pair", 0) == 0 and not r.get("error")]
print(f"T0 {datetime.fromtimestamp(t0, timezone.utc):%H:%M:%S} UTC; scrape samples {len(samples)}; first attempts {len(first)}")
hdr = ("win(min)", "gen/s", "prompt/s", "GPU-hit%", "CPU-rest%", "recomp%", "KV", "run/wait", "holds", "preempt",
       "1st done", "1st corr")
print(" | ".join(hdr))
if not samples:
    sys.exit()
start = samples[0]["t"]
end = samples[-1]["t"]
k = 0
while start + (k + 1) * B <= end + 1:
    a, b = start + k * B, start + (k + 1) * B
    sa = min(samples, key=lambda s: abs(s["t"] - a))
    sb = min(samples, key=lambda s: abs(s["t"] - b))
    dt = max(1, sb["t"] - sa["t"])
    d = lambda key: sb.get(key, 0) - sa.get(key, 0)
    q, h, eh = d("prefix_cache_queries_total"), d("prefix_cache_hits_total"), d("external_prefix_cache_hits_total")
    gpu = 100 * h / q if q else 0
    cpu = 100 * eh / q if q else 0
    rec = 100 - gpu - cpu if q else 0
    inwin = [s for s in samples if a <= s["t"] <= b]
    kv = max((s.get("kv_cache_usage_perc", 0) for s in inwin), default=0)
    runw = f"{max((s.get('num_requests_running', 0) for s in inwin), default=0):.0f}/" \
           f"{max((s.get('num_requests_waiting', 0) for s in inwin), default=0):.0f}"
    ea, eb = a - t0, b - t0
    nh = sum(1 for x in holds if ea <= x < eb)
    fa = [r for r in first if r.get("t_end_h") is not None and r["t_end_h"] * 3600 <= eb]
    fc = sum(1 for r in fa if correct(r))
    print(" | ".join(str(x) for x in (f"{k * B // 60}-{(k + 1) * B // 60}", round(d("generation_tokens_total") / dt),
                                       round(d("prompt_tokens_total") / dt), f"{gpu:.1f}", f"{cpu:.1f}", f"{rec:.1f}",
                                       f"{kv:.2f}", runw, nh, int(d("num_preemptions_total")), len(fa), fc)))
    k += 1
