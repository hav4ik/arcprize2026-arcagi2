"""Dense GPU memory series from a vLLM server log: per bin, the lowest vLLM-reported free and cap minus peak over ranks.
usage: gpumem_series.py LOG T0_UTC(HH:MM:SS) [BIN_MIN]"""
import re, sys, datetime as dt
log, t0s = sys.argv[1], sys.argv[2]
bin_s = 60 * float(sys.argv[3] if len(sys.argv) > 3 else 10)
ts = r"(\d\d-\d\d \d\d:\d\d:\d\d)"
cap_re = re.compile(r"Worker_TP(\d).*?" + ts + r".*?Capped the PyTorch allocator at ([\d.]+) GiB")
mem_re = re.compile(r"Worker_TP(\d).*?" + ts + r".*?GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved.*?, ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs")
caps, rows = {}, []
for line in open(log, errors="replace"):
    m = cap_re.search(line)
    if m:
        caps[int(m[1])] = float(m[3]); continue
    m = mem_re.search(line)
    if m and int(m[1]) in caps:
        t = dt.datetime.strptime("2026-" + m[2], "%Y-%m-%d %H:%M:%S")
        rows.append((t, int(m[1]), caps[int(m[1])] - float(m[4]), float(m[6]), int(m[7]), int(m[8]), caps[int(m[1])] - float(m[5])))
day = rows[0][0].date()
t0 = dt.datetime.combine(day, dt.time.fromisoformat(t0s))
bins = {}
for t, r, cp, free, retr, oom, cr in rows:
    b = int((t - t0).total_seconds() // bin_s)
    if b < 0: continue
    v = bins.setdefault(b, [9e9, 9e9, 0, 0, t, 9e9])
    v[0] = min(v[0], cp); v[1] = min(v[1], free); v[5] = min(v[5], cr); v[2] = max(v[2], retr); v[3] = max(v[3], oom); v[4] = t
for b, (cp, free, retr, oom, t, cr) in sorted(bins.items()):
    th = (t - t0).total_seconds() / 3600
    print(f"{th:.2f}\t{cp:.2f}\t{free:.2f}\t{retr}\t{oom}\t{t:%H:%M}\t{cr:.2f}")
