"""Throughput series from a vLLM server log's 10 s 'Avg generation throughput' lines: per bin, mean generated tok/s,
mean running / waiting requests and mean KV use. usage: tps_series.py LOG T0_UTC(HH:MM:SS) [BIN_MIN]"""
import re, sys, datetime as dt
log, t0s = sys.argv[1], sys.argv[2]
bin_s = 60 * float(sys.argv[3] if len(sys.argv) > 3 else 10)
pat = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*?Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
                 r"Waiting: (\d+) reqs,.*?GPU KV cache usage: ([\d.]+)%")
rows = []
for line in open(log, errors="replace"):
    m = pat.search(line)
    if m:
        rows.append((dt.datetime.strptime("2026-" + m[1], "%Y-%m-%d %H:%M:%S"), float(m[2]), int(m[3]), int(m[4]), float(m[5])))
if not rows:
    sys.exit(0)
t0 = dt.datetime.combine(rows[0][0].date(), dt.time.fromisoformat(t0s))
bins = {}
for t, g, r, w, kv in rows:
    b = int((t - t0).total_seconds() // bin_s)
    if b >= 0:
        bins.setdefault(b, []).append((t, g, r, w, kv))
for b, v in sorted(bins.items()):
    n = len(v)
    th = ((v[0][0] - t0).total_seconds() + (v[-1][0] - t0).total_seconds()) / 2 / 3600
    print(f"{th:.2f}\t{sum(x[1] for x in v)/n:.1f}\t{sum(x[2] for x in v)/n:.1f}\t{sum(x[3] for x in v)/n:.1f}\t{sum(x[4] for x in v)/n:.1f}\t{n}")
