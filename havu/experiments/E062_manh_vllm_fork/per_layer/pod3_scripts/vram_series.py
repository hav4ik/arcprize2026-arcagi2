"""Per-process GPU memory (nvidia-smi compute apps, max over the 4 workers) and host AnonPages from restart_sampler.py
samples, the first sample of every <bucket> s; marks the vLLM pid set so restarts show.
Usage: vram_series.py <sampler.jsonl> [bucket_s=600]"""
import json, sys

S = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 600
seen, prev = set(), None
for r in S:
    k = int(r["t"] // B)
    if k in seen:
        continue
    seen.add(k)
    g = [int(x.split(",")[1]) for x in r["gpu"] if "," in x]
    pids = tuple(sorted(p for p, _, _ in r["vllm"]))
    mark = "" if pids == prev or prev is None else "  <- new pid set"
    prev = pids
    print(f"{r['utc']}  GPU app MiB max {max(g) if g else '-':>6}  min {min(g) if g else '-':>6}  "
          f"AnonPages {r['mem_kb']['AnonPages'] / 1048576:6.1f} GiB  all RSS {r['all_rss_kb'] / 1048576:6.1f} GiB{mark}")
