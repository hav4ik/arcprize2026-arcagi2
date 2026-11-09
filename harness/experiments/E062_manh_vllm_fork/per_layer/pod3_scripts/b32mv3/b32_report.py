"""B32-M v3 replica report from a local copy: per-hour (Kaggle clock = solver elapsed) host RAM (AnonPages, Shmem,
Kaggle-eq = AnonPages + Shmem + 7 GiB, /dev/shm used, sandboxes) and VRAM (min free per GPU) from mon.jsonl; the vLLM
"Capped the PyTorch allocator" lines and the newest "GPU memory:" line per rank (allocated, peak, reserved, split,
free, retries, OOMs; cap - peak) from the server logs; solver deaths / restarts; error records.
Usage: b32_report.py <copy dir: holds dsv41_logs/, mon.jsonl>"""
import glob, json, os, re, sys

D = sys.argv[1]; G = 1048576
M = [json.loads(l) for l in open(f"{D}/mon.jsonl") if l.strip() and '"meminfo_kb"' in l]
sl = open(f"{D}/dsv41_logs/solver.log", errors="replace").read() if os.path.exists(f"{D}/dsv41_logs/solver.log") else ""
m0 = re.search(r"\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\]", sl)
t0 = None
if m0 and M:
    import datetime as dt
    day = dt.datetime.fromtimestamp(M[0]["t"], dt.timezone.utc).replace(hour=int(m0[1]), minute=int(m0[2]), second=int(m0[3]))
    t0 = day.timestamp() - float(m0[4]) * 3600
    if t0 > M[0]["t"] + 3600: t0 -= 86400
print("| Kaggle h | UTC | AnonPages | Shmem | Kaggle-eq | /dev/shm | sandboxes n / GiB / max | engines | min GPU free MiB |")
print("|---|---|---|---|---|---|---|---|---|")
seen = set()
for r in M:
    h = (r["t"] - t0) / 3600 if t0 else None
    k = int(h * 2) if h is not None else int(r["t"] // 1800)  # every 30 min
    if k in seen: continue
    seen.add(k)
    mi = r["meminfo_kb"]; g = [x for x in r.get("gpu", []) if isinstance(x, list)]; sb = r.get("sandboxes", {})
    print(f"| {h:.2f} | {r['utc']} | {mi['AnonPages'] / G:.1f} | {mi['Shmem'] / G:.1f} | {r['kaggle_eq_gib']:.1f} | "
          f"{r['dev_shm_used_kb'] / G:.1f} | {sb.get('n', 0)} / {sb.get('total_kb', 0) / G:.1f} / {sb.get('max_kb', 0) / G:.2f} | "
          f"{r.get('engines')} | {int(min(x[2] for x in g)) if g else '-'} |")
pk = max(M, key=lambda r: r["kaggle_eq_gib"]) if M else None
if pk: print(f"\npeak Kaggle-eq {pk['kaggle_eq_gib']:.1f} GiB at {pk['utc']}; min GPU free {min(min(x[2] for x in r['gpu'] if isinstance(x, list)) for r in M if r.get('gpu')):.0f} MiB")
logs = sorted(glob.glob(f"{D}/dsv41_logs/**/server*.log", recursive=True))
cap, last = {}, {}
for f in logs:
    for line in open(f, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        w = re.search(r"\((Worker\S*|EngineCore\S*) pid=(\d+)\)", line); who = w[1] if w else "?"
        c = re.search(r"Capped the PyTorch allocator at ([\d.]+) GiB \(([\d.]+) GiB reserved\)", line)
        if c: cap[who] = (float(c[1]), float(c[2]), os.path.basename(f))
        m = re.search(r"GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved \(([\d.]+) GiB in split segments\), ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs", line)
        if m: last[who] = [float(x) for x in m.groups()[:5]] + [int(m[6]), int(m[7]), line[:40].strip(), os.path.basename(f)]
print(f"\nserver logs: {[os.path.basename(x) for x in logs]}")
for who in sorted(set(cap) | set(last)):
    c = cap.get(who); m = last.get(who)
    cs = f"cap {c[0]:.2f} (reserved {c[1]:.2f}, cap-reserved {c[0] - c[1]:.2f})" if c else "cap -"
    ms = (f"alloc {m[0]:.2f} peak {m[1]:.2f} reserved {m[2]:.2f} split {m[3]:.2f} free {m[4]:.2f} retries {m[5]} OOMs {m[6]}"
          + (f"; cap-peak {c[0] - m[1]:.2f}" if c else "")) if m else "no GPU memory line"
    print(f"  {who}: {cs}; {ms}")
for pat in ("server died", "server up after", "restart", "out of memory", "OutOfMemoryError"):
    hits = [l for l in sl.splitlines() if pat in l]
    print(f"solver.log '{pat}': {len(hits)}" + (f"; last: {hits[-1][:200]}" if hits else ""))
R = [json.loads(l) for l in open(f"{D}/dsv41_logs/attempts.jsonl")] if os.path.exists(f"{D}/dsv41_logs/attempts.jsonl") else []
import collections
print(f"attempts.jsonl: {len(R)} records, {sum(1 for r in R if r.get('error'))} with error: {dict(collections.Counter(r['error'][:60] for r in R if r.get('error')))}")
