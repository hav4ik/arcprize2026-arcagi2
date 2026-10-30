"""Hourly report for the Manh v12 replica from v12_sampler.py's jsonl + the solver's files (read-only).
Usage: v12_report.py <samp.jsonl> <dsv41_logs dir> <solutions.json> [pool]"""
import collections, json, subprocess, sys
S, L, SOL = sys.argv[1:4]; pool = sys.argv[4] if len(sys.argv) > 4 else ""
rs = [json.loads(l) for l in open(S) if l.strip()]
rs = [r for r in rs if "meminfo_gib" in r]
t0 = rs[0]["t"]; byh = collections.OrderedDict()
for r in rs:
    byh.setdefault(int((r["t"] - t0) // 1800), []).append(r)  # 30-min buckets
print("| UTC (30-min bucket end) | Kaggle-eq max | AnonPages max | Shmem max | MemAvail min | RSS TP / Eng / API / sup / solver / sandbox / kernel (max) | nvidia free min per GPU (GiB) | vLLM alloc / peak / reserved / free (last, worst rank) |")
print("|---|---|---|---|---|---|---|---|")
for h, b in byh.items():
    m = lambda k: max(x["meminfo_gib"].get(k, 0) for x in b)
    rss = {k: max(x.get("rss_gib", {}).get(k, 0) for x in b) for k in ("tp_workers", "engine_core", "api_server", "supervisor", "solver", "sandboxes", "kernel")}
    fr = {}
    for x in b:
        for g in x.get("gpus", []):
            if g["used_mib"] > 4000: fr[g["gpu"]] = min(fr.get(g["gpu"], 99), g["free_mib"] / 1024)
    v = b[-1].get("vllm_gpu_mem") or {}
    w = min(v.values(), key=lambda q: q["free"]) if v else None
    print(f"| {b[-1]['utc']} | {max(x.get('kaggle_eq_gib', 0) for x in b):.1f} | {m('AnonPages'):.1f} | {m('Shmem'):.1f} | "
          f"{min(x['meminfo_gib']['MemAvailable'] for x in b):.1f} | " + " / ".join(f"{rss[k]:.1f}" for k in rss) + " | "
          + " ".join(f"{fr[g]:.2f}" for g in sorted(fr)) + " | "
          + (f"{w['alloc']:.2f} / {w['peak']:.2f} / {w['reserved']:.2f} / {w['free']:.2f} (retries {sum(q['retries'] for q in v.values())}, OOMs {sum(q['ooms'] for q in v.values())})" if w else "-") + " |")
last = rs[-1]
print("\nallocator cap per rank (startup 'Capped the PyTorch allocator at X (Y reserved)'), from the last sample of each 30-min bucket:")
print("| UTC | " + " | ".join(f"TP{i}: X-Y / cap-peak / cap-reserved / retries" for i in range(4)) + " |")
print("|---|" + "---|" * 4)
for h, b in byh.items():
    c = b[-1].get("alloc_cap") or {}; v = b[-1].get("vllm_gpu_mem") or {}
    if c:
        print(f"| {b[-1]['utc']} | " + " | ".join(f"{c[k]['cap_minus_reserved0']:.2f} / {c[k]['cap_minus_peak']:.2f} / {c[k]['cap_minus_reserved_now']:.2f} / {v.get(k, {}).get('retries')}" if k in c and c[k]['cap_minus_peak'] is not None else "-" for k in ("TP0", "TP1", "TP2", "TP3")) + " |")
print(f"\nlast sample {last['utc']}: Kaggle-eq {last.get('kaggle_eq_gib')} GiB; supervisor {json.dumps(last.get('supervisor'))[:300]}; solver {json.dumps(last.get('solver'))[:400]}")
att = [json.loads(l) for l in open(f"{L}/attempts.jsonl")] if __import__('os').path.exists(f"{L}/attempts.jsonl") else []
er = [a for a in att if a.get("error") or a.get("stop_reason") == "error"]
print(f"attempt records {len(att)}; with error {len(er)}: {dict(collections.Counter(str(a.get('error'))[:100] for a in er))}")
print(subprocess.run([sys.executable, "-I", "/root/e062pl/first_correct.py", f"{L}/attempts.jsonl", SOL], capture_output=True, text=True).stdout.strip())
print(subprocess.run(f"{sys.executable} -I /root/e062pl/windows2.py {L}/metrics.jsonl {pool} | tail -14", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run(f"grep -aiE 'server died|wait [0-9]\\)|server up after|KV gate' {L}/solver.log | tail -6 | cut -c1-220", shell=True, capture_output=True, text=True).stdout)
# slopes over the last 2 h (least squares on 60 s samples) and time to each limit at that slope
def fit(pts):
    if len(pts) < 10: return None
    n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    return (sum((x - mx) * (y - my) for x, y in pts) / sxx) if sxx else None
t_up = next((r["t"] for r in rs if r.get("rss_gib", {}).get("tp_workers", 0) > 100), rs[0]["t"])  # engine loaded
w = [r for r in rs if r["t"] >= max(rs[-1]["t"] - 7200, t_up + 1200)]  # last 2 h, excluding the first 20 min of load
def series(f):
    out = []
    for r in w:
        try:
            v = f(r)
            if v is not None: out.append(((r["t"] - w[0]["t"]) / 3600, v))
        except Exception:
            pass
    return out
def worst(r, key):
    c = r.get("alloc_cap") or {}
    vals = [v[key] for v in c.values() if v.get(key) is not None]
    return min(vals) if vals else None
def vfree(r):
    vals = [g["free_mib"] / 1024 for g in r.get("gpus", []) if g["used_mib"] > 4000]
    return min(vals) if vals else None
print("\nslopes over the last %.1f h (per hour) and time to limit at that slope:" % ((w[-1]["t"] - w[0]["t"]) / 3600))
for name, f, limit, down in (("Kaggle-eq RAM GiB", lambda r: r.get("kaggle_eq_gib"), 173.0, False),
                             ("worst cap - peak GiB", lambda r: worst(r, "cap_minus_peak"), 0.26, True),
                             ("worst cap - reserved GiB", lambda r: worst(r, "cap_minus_reserved_now"), 0.0, True),
                             ("nvidia-smi free min GiB", vfree, 1.0, True),
                             ("sandboxes RSS GiB", lambda r: r.get("rss_gib", {}).get("sandboxes"), None, False),
                             ("TP workers RSS GiB", lambda r: r.get("rss_gib", {}).get("tp_workers"), None, False)):
    s_ = series(f); k = fit(s_)
    if k is None or not s_:
        print(f"  {name}: n/a"); continue
    cur = s_[-1][1]; eta = ""
    if limit is not None and ((down and k < 0) or (not down and k > 0)):
        eta = f"; reaches {limit} in ~{abs((cur - limit) / k):.1f} h"
    elif limit is not None:
        eta = "; not moving toward the limit"
    print(f"  {name}: now {cur:.2f}, slope {k:+.3f}/h{eta}")
print(subprocess.run(f"grep -acE 'APITimeoutError|timed out' {L}/solver.log", shell=True, capture_output=True, text=True).stdout.strip(), "solver.log lines with APITimeoutError / timed out")
