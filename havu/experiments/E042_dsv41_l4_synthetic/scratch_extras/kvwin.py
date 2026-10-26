import sys
rows = []
for l in open(sys.argv[1]):
    p = l.split()
    t = p[0]
    if not ("17:25:00" <= t <= "18:59:59"):
        continue
    d = {}
    for kv in p[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                d.setdefault(k, float(v))
            except ValueError:
                pass
    h, m, s = map(int, t.split(":"))
    d["ts"] = h * 3600 + m * 60 + s
    d["t"] = t
    rows.append(d)


def win(a, b, label):
    r = [x for x in rows if a <= x["t"] <= b]
    x0, x1 = r[0], r[-1]
    dt = x1["ts"] - x0["ts"]
    g = (x1["generation_tokens_total"] - x0["generation_tokens_total"]) / dt
    q = x1["prefix_cache_queries_total"] - x0["prefix_cache_queries_total"]
    hh = x1["prefix_cache_hits_total"] - x0["prefix_cache_hits_total"]
    kv = sorted(x["kv_cache_usage_perc"] for x in r)
    run = [x["num_requests_running"] for x in r]
    wt = [x["num_requests_waiting"] for x in r]
    pre = x1["num_preemptions_total"] - x0["num_preemptions_total"]
    print(f"{label} {x0['t']}-{x1['t']}: gen {g:6.1f} tok/s  hit {hh / max(q, 1):.3f} (queried {q / dt:6.0f} tok/s, "
          f"uncached {(q - hh) / dt:5.0f} tok/s)  KV p50 {kv[len(kv) // 2]:.2f} max {kv[-1]:.2f}  "
          f"running {min(run):.0f}-{max(run):.0f}  waiting max {max(wt):.0f}  preemptions {pre:.0f}")


for a, b in [("17:25", "17:40"), ("17:40", "18:00"), ("18:00", "18:20"), ("18:20", "18:40"), ("18:40", "18:59:59")]:
    win(a, b, "bin")
win("17:45", "18:59:59", "STEADY")
win("17:25", "18:59:59", "ALL")
