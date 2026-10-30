"""ram.tsv for the dashboard: the sampler's Kaggle-eq host RAM, 1 point per 10 min (the first sample of each 10-min
bucket, plus the newest). Kaggle-eq = cgroup v1 rss + shmem + 7 GiB; samples before 03:50 UTC (no cgroup field) use the
processes' summed RssAnon + Shmem + 7 (this pod's /proc/meminfo is host-wide). Usage: ram_tsv.py <mon.jsonl> <T0 epoch>"""
import json, sys, time
T0 = float(sys.argv[2]); G = 1048576
R = [json.loads(l) for l in open(sys.argv[1]) if '"meminfo_kb"' in l]
print("utc\tkaggle_clock_h\tkaggle_eq_gib")
seen = set()
for i, r in enumerate(R):
    k = int(r["t"] // 600)
    if k in seen and i != len(R) - 1:
        continue
    seen.add(k)
    cg = r.get("cgroup_v1_bytes") or {}
    keq = (cg["total_rss"] + cg["total_shmem"]) / 2**30 + 7 if "total_rss" in cg else \
        (sum(g["anon_kb"] for g in r["groups"].values()) + r["meminfo_kb"]["Shmem"]) / G + 7
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(r['t']))}\t{(r['t'] - T0) / 3600:.3f}\t{keq:.2f}")
