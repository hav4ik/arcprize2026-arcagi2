"""B32-M v3 replica trends from a local copy (pull_b32.sh): a few points per hour of the Kaggle clock (solver elapsed)
for VRAM per rank (peak allocated, reserved, cap - peak, allocator retries, OOMs, free; from the server's 60 s
"GPU memory:" lines) and host memory (AnonPages, Shmem, Kaggle-eq, sandboxes; mon.jsonl), the slope over the last hour,
and a time-to-limit estimate (VRAM: cap - peak reaching 0.26 GiB, the auditor's OOM band; host: Kaggle-eq 173 GiB).
Usage: b32_trends.py <copy dir> [points per hour=2]"""
import glob, json, os, re, sys, datetime as dt

D = sys.argv[1]; PPH = int(sys.argv[2]) if len(sys.argv) > 2 else 2; G = 1048576
sl = open(f"{D}/dsv41_logs/solver.log", errors="replace").read()
m0 = re.search(r"\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\]", sl)
M = [json.loads(l) for l in open(f"{D}/mon.jsonl") if '"meminfo_kb"' in l]
base = dt.datetime.fromtimestamp(M[0]["t"], dt.timezone.utc).replace(hour=int(m0[1]), minute=int(m0[2]), second=int(m0[3]))
T0 = base.timestamp() - float(m0[4]) * 3600
if T0 > M[0]["t"] + 3600: T0 -= 86400
kh = lambda t: (t - T0) / 3600


def keq(r):
    """Kaggle-eq of this container: cgroup v1 rss + shmem + 7 GiB (the sampler's field from 03:50 UTC); older records:
    the processes' summed RssAnon + Shmem + 7 (this pod's /proc/meminfo is host-wide: ~12-13 GiB of foreign anon)."""
    cg = r.get("cgroup_v1_bytes") or {}
    if "total_rss" in cg:
        return (cg["total_rss"] + cg["total_shmem"]) / 2**30 + 7
    return (sum(g["anon_kb"] for g in r["groups"].values()) + r["meminfo_kb"]["Shmem"]) / G + 7


def anon(r):
    cg = r.get("cgroup_v1_bytes") or {}
    return cg["total_rss"] / 2**30 if "total_rss" in cg else sum(g["anon_kb"] for g in r["groups"].values()) / G
# VRAM series per rank: (t, alloc, peak, reserved, split, free, retries, ooms); cap per rank per server start
pat = re.compile(r"\((Worker_TP(\d)\S*) pid=(\d+)\) INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved \(([\d.]+) GiB in split segments\), ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs")
cpat = re.compile(r"\((Worker_TP(\d)\S*) pid=(\d+)\).*Capped the PyTorch allocator at ([\d.]+) GiB \(([\d.]+) GiB reserved\)")
V, cap = {}, {}
year = dt.datetime.fromtimestamp(M[0]["t"], dt.timezone.utc).year
for f in sorted(glob.glob(f"{D}/dsv41_logs/**/server*.log", recursive=True), key=os.path.getmtime):
    for line in open(f, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        c = cpat.search(line)
        if c: cap[(int(c[2]), c[3])] = float(c[4])
        m = pat.search(line)
        if m:
            t = dt.datetime(year, int(m[4]), int(m[5]), int(m[6]), int(m[7]), int(m[8]), tzinfo=dt.timezone.utc).timestamp()
            V.setdefault(int(m[2]), []).append((t, m[3], *[float(x) for x in m.groups()[8:13]], int(m[14]), int(m[15])))
print("VRAM per rank (GiB): Kaggle h | rank | peak | reserved | cap-peak | retries | OOMs | free")
for r in sorted(V):
    seen = set()
    for t, pid, a, pk, rs, sp, fr, rt, oo in V[r]:
        k = int(kh(t) * PPH)
        if k in seen and (t, pid) != (V[r][-1][0], V[r][-1][1]): continue
        seen.add(k); c = cap.get((r, pid))
        print(f"  {kh(t):5.2f} | TP{r} | {pk:.2f} | {rs:.2f} | {c - pk if c else float('nan'):.2f} | {rt} | {oo} | {fr:.2f}")
    # slope of cap - peak over the last hour of this rank's newest server
    last = V[r][-1]; c = cap.get((r, last[1]))
    hr = [x for x in V[r] if x[1] == last[1] and x[0] >= last[0] - 3600]
    # tuple: (t, pid, alloc, peak, reserved, split, free, retries, ooms); skip the first 10 min after server up
    hr = [x for x in hr if x[0] >= V[r][0][0] + 600] or hr
    if c and len(hr) > 2 and hr[-1][0] > hr[0][0]:
        span = (hr[-1][0] - hr[0][0]) / 3600
        sl_h = (hr[-1][3] - hr[0][3]) / span  # peak GiB/h
        left = c - last[3] - 0.26
        eta = f"{left / sl_h:.1f} h" if sl_h > 1e-4 else "no growth"
        print(f"  TP{r}: peak {sl_h:+.3f} GiB/h, reserved {(hr[-1][4] - hr[0][4]) / span:+.3f} GiB/h (over {span:.2f} h); "
              f"cap-peak {c - last[3]:.2f}; to cap-peak 0.26 at this slope: {eta}; retries {last[7]} (+{hr[-1][7] - hr[0][7]}), OOMs {last[8]}")
print("host (GiB): Kaggle h | container anon | Shmem | Kaggle-eq | /dev/shm | sandboxes n / total / max | min GPU free MiB")
seen = set()
for r in M:
    k = int(kh(r["t"]) * PPH)
    if k in seen and r is not M[-1]: continue
    seen.add(k); mi = r["meminfo_kb"]; sb = r.get("sandboxes", {}); g = [x for x in r.get("gpu", []) if isinstance(x, list)]
    print(f"  {kh(r['t']):5.2f} | {anon(r):.1f} | {mi['Shmem'] / G:.1f} | {keq(r):.1f} | {r['dev_shm_used_kb'] / G:.1f} | "
          f"{sb.get('n', 0)} / {sb.get('total_kb', 0) / G:.1f} / {sb.get('max_kb', 0) / G:.2f} | {int(min(x[2] for x in g)) if g else '-'}")
steady = [r for r in M if r["meminfo_kb"]["Shmem"] / G > 100]  # server loaded (Engram + offload in shmem)
hr = [r for r in steady if r["t"] >= M[-1]["t"] - 3600 and r["t"] >= steady[0]["t"] + 600] if steady else []
if len(hr) > 2:
    s = (keq(hr[-1]) - keq(hr[0])) / ((hr[-1]["t"] - hr[0]["t"]) / 3600)
    print(f"  Kaggle-eq {keq(hr[-1]):.1f} GiB, {s:+.2f} GiB/h over the last hour; to 173: "
          + (f"{(173 - keq(hr[-1])) / s:.1f} h" if s > 0.01 else "no growth")
          + f"; peak {max(keq(x) for x in M):.1f}; engines now {M[-1].get('engines')}; max engines {max(x.get('engines') or 0 for x in M)}")
