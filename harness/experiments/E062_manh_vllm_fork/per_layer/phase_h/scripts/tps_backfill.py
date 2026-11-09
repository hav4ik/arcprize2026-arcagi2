"""One-time backfill of tps.tsv rows (per minute) from the server log's 10 s "Engine 000" lines, from the notebook
start until <until utc HH:MM:SS> (the first live /metrics row). Usage: tps_backfill.py <server.log> <T_nb epoch> <until HH:MM:SS> <out>"""
import calendar, re, sys, time
log, tnb, until, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
E = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers.py:\d+\] Engine 000: Avg prompt throughput: [\d.]+ tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, .*?GPU KV cache usage: ([\d.]+)%")  # lines may carry "KV fetch: …, …," before the KV usage
yr = time.gmtime(tnb).tm_year; d0 = time.strftime("%Y-%m-%d", time.gmtime(tnb))
tu = calendar.timegm(time.strptime(f"{d0} {until}", "%Y-%m-%d %H:%M:%S"))
mins = {}
for l in open(log, errors="replace"):
    m = E.search(ANSI.sub("", l))
    if not m: continue
    mo, d, H, M, S = map(int, m.groups()[:5]); t = calendar.timegm((yr, mo, d, H, M, S))
    if t >= tu: continue
    mins.setdefault(t // 60, []).append((float(m.group(6)), int(m.group(7)), int(m.group(8)), float(m.group(9))))
with open(out, "w") as f:
    for k in range(int(tnb // 60) + 1, int(tu // 60)):
        x = mins.get(k); t = k * 60
        row = (f"{sum(a for a, *_ in x) / len(x):.1f}\t{round(sum(b for _, b, _, _ in x) / len(x))}\t{round(sum(c for *_, c, _ in x) / len(x))}\t{sum(q for *_, q in x) / len(x):.1f}"
               if x else "0\t0\t0\t")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))}\t{(t - tnb) / 3600:.3f}\t{row}\n")
print("backfill rows", int(tu // 60) - int(tnb // 60) - 1)
