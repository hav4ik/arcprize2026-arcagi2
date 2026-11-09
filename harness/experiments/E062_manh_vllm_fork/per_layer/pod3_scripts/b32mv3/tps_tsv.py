"""tps.tsv for the dashboard: one row per complete minute from the first metrics scrape. Each scrape pair (p, s) of
the 60 s scraper gives gen_tok_s = delta vllm:generation_tokens_total / gap (a counter reset at a restart counts from
0), and fills every minute in (p, s] with that rate and s's running / waiting / kv_pct, so scrape-timing jitter (a
minute with no or two scrapes) never looks like an outage. Minutes inside a server-down window from the solver log
("server died" -> "server up after"), or after a scrape gap > 180 s (/metrics unreachable), get 0 / 0 / 0 and an
empty kv_pct. Rows stop at the newest scraped minute (no in-progress minute), except that a server that is down now
gets zero rows up to the last complete minute. Usage: tps_tsv.py <metrics_scrape.txt> <T0 epoch>"""
import calendar, re, sys, time
T0 = float(sys.argv[2])
day = time.strftime("%Y-%m-%d", time.gmtime(T0))
S, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if line.startswith("T "):
        cur = {"t": int(line.split()[1])}; S.append(cur)
    elif cur is not None and line.startswith("vllm:"):
        k, v = line.rsplit(" ", 1)
        try:
            cur[k.replace("vllm:", "")] = cur.get(k.replace("vllm:", ""), 0.0) + float(v)
        except ValueError:
            pass
S = [s for s in S if "generation_tokens_total" in s]
down, died = [], None  # server-down windows from the solver log
try:
    for line in open("/kaggle/working/dsv41_logs/solver.log", errors="replace"):
        m = re.match(r"\[(\d\d:\d\d:\d\d) \+([\d.]+)h\]", line)
        if not m:
            continue
        t0 = T0 + float(m[2]) * 3600
        t = min((calendar.timegm(time.strptime(f"{day} {m[1]}", "%Y-%m-%d %H:%M:%S")) + k * 86400 for k in (-1, 0, 1)), key=lambda x: abs(x - t0))
        if "server died" in line:
            died = t
        elif "server up after" in line and died is not None:
            down.append((died, t)); died = None
except OSError:
    pass
now_down = died is not None
isdown = lambda t: any(a <= t <= b for a, b in down) or (now_down and t >= died)
rows = {}
for i in range(1, len(S)):
    p, s = S[i - 1], S[i]; dt = s["t"] - p["t"]
    for m in range(p["t"] // 60 + 1, s["t"] // 60 + 1):
        if dt > 180 or isdown(m * 60):
            rows[m] = None
        else:
            d = s["generation_tokens_total"] - p["generation_tokens_total"]
            rows[m] = ((d if d >= 0 else s["generation_tokens_total"]) / dt, s)
print("utc\tkaggle_h\tgen_tok_s\trunning\twaiting\tkv_pct")
if S:
    last = S[-1]["t"] // 60
    if now_down:
        last = max(last, int(time.time()) // 60 - 1)
    for m in range(S[0]["t"] // 60 + 1, last + 1):
        t = m * 60; v = rows.get(m)
        head = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))}\t{(t - T0) / 3600:.3f}"
        if v is None:
            print(f"{head}\t0\t0\t0\t")
        else:
            r, s = v
            print(f"{head}\t{r:.1f}\t{int(s.get('num_requests_running', 0))}\t{int(s.get('num_requests_waiting', 0))}\t{100 * s.get('kv_cache_usage_perc', 0):.1f}")
