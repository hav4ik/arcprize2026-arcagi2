import sys
# usage: kvwin2.py kvmon.log HH:MM:SS(start) [bin_minutes]
start = sys.argv[2]
binm = float(sys.argv[3]) if len(sys.argv) > 3 else 5
rows = []
for l in open(sys.argv[1]):
    p = l.split()
    t = p[0]
    if t < start:
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


def win(r, label):
    if len(r) < 2:
        return
    x0, x1 = r[0], r[-1]
    dt = x1["ts"] - x0["ts"]
    g = (x1["generation_tokens_total"] - x0["generation_tokens_total"]) / dt
    q = x1["prefix_cache_queries_total"] - x0["prefix_cache_queries_total"]
    hh = x1["prefix_cache_hits_total"] - x0["prefix_cache_hits_total"]
    kv = sorted(x["kv_cache_usage_perc"] for x in r)
    run = [x["num_requests_running"] for x in r]
    wt = [x["num_requests_waiting"] for x in r]
    pre = x1["num_preemptions_total"] - x0["num_preemptions_total"]
    print(f"{label} {x0['t']}-{x1['t']}: gen {g:6.1f} tok/s  hit {hh / max(q, 1):.3f}  uncached {(q - hh) / dt:5.0f} tok/s  "
          f"KV p50 {kv[len(kv) // 2]:.2f} max {kv[-1]:.2f}  running {min(run):.0f}-{max(run):.0f}  "
          f"waiting max {max(wt):.0f}  preempt {pre:.0f}")


t0 = rows[0]["ts"] if rows else 0
b = []
for x in rows:
    if b and x["ts"] - b[0]["ts"] >= binm * 60:
        win(b + [x], "bin")
        b = []
    b.append(x)
win(b, "bin")
win(rows, "ALL")
