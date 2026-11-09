"""Restart timeline from restart_sampler.py samples (5 s): the old engine's pids (present at the kill), when each is
gone, when the GPU process list is empty, when new vLLM pids appear and the new GPU apps, the supervisor state changes,
whether old and new engines ever coexist, and host RAM (AnonPages, all-process RSS, MemAvailable) peaks in the window.
Usage: analyze_restart.py <sampler.jsonl> <kill epoch> [window_s=3600]"""
import json, sys, time

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tk = float(sys.argv[2])
win = float(sys.argv[3]) if len(sys.argv) > 3 else 3600
G = 1048576
before = [r for r in recs if r["t"] <= tk]
after = [r for r in recs if tk < r["t"] <= tk + win]
old = {p for p, _, _ in before[-1]["vllm"]}
old_gpu = {g.split(",")[0].strip() for g in before[-1]["gpu"]}
fmt = lambda t: time.strftime("%H:%M:%S", time.gmtime(t))
print(f"kill at {fmt(tk)}; old engine pids {sorted(old)}; last sample before kill {before[-1]['utc']}: "
      f"AnonPages {before[-1]['mem_kb']['AnonPages'] / G:.1f} GiB, all RSS {before[-1]['all_rss_kb'] / G:.1f} GiB")
ev, gone, gpu_empty, new_first, coexist, last_state = {}, {}, None, None, [], None
for r in after:
    cur = {p for p, _, _ in r["vllm"]}
    for p in old - cur:
        gone.setdefault(p, r["t"])
    if gpu_empty is None and not r["gpu"]:
        gpu_empty = r["t"]
    new = cur - old
    if new and new_first is None:
        new_first = r["t"]
    if new and (cur & old):
        coexist.append((r["utc"], sorted(cur & old), sorted(new)))
    st = r["status"].get("e066_supervisor.json")
    try:
        s = json.loads(st).get("state") if st else None
    except ValueError:
        s = "unparsable"
    if s != last_state:
        print(f"  {r['utc']} (+{r['t'] - tk:.0f}s) supervisor state -> {s}")
        last_state = s
for p in sorted(old):
    print(f"  old pid {p} gone at {fmt(gone[p]) + f' (+{gone[p] - tk:.0f}s)' if p in gone else 'NOT GONE'}")
print(f"  GPU process list empty at {fmt(gpu_empty) + f' (+{gpu_empty - tk:.0f}s)' if gpu_empty else 'never'}")
print(f"  first new vLLM pid at {fmt(new_first) + f' (+{new_first - tk:.0f}s)' if new_first else 'none'}")
print(f"  old and new engines coexisting: {coexist[:3] if coexist else 'never'}")
if after:
    pk = max(after, key=lambda r: r["mem_kb"]["AnonPages"])
    pr = max(after, key=lambda r: r["all_rss_kb"])
    mn = min(after, key=lambda r: r["mem_kb"]["MemAvailable"])
    lo = min(after, key=lambda r: r["mem_kb"]["AnonPages"])
    print(f"  window peak AnonPages {pk['mem_kb']['AnonPages'] / G:.1f} GiB at {pk['utc']}; peak all-process RSS "
          f"{pr['all_rss_kb'] / G:.1f} GiB at {pr['utc']}; min MemAvailable {mn['mem_kb']['MemAvailable'] / G:.1f} GiB at {mn['utc']}; "
          f"lowest AnonPages {lo['mem_kb']['AnonPages'] / G:.1f} GiB at {lo['utc']}")
    print(f"  last sample {after[-1]['utc']}: AnonPages {after[-1]['mem_kb']['AnonPages'] / G:.1f} GiB, vLLM pids {[p for p, _, _ in after[-1]['vllm']]}, GPU {after[-1]['gpu']}")
