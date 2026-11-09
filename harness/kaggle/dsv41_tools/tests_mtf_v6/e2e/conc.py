"""Max / time-averaged number of concurrently running attempts, from attempts.jsonl (start = t_end - wall_s)."""
import json, sys
for R in sys.argv[1:]:
    iv = []
    for l in open(f"{R}/work/dsv41_logs/attempts.jsonl"):
        r = json.loads(l)
        if r.get("t_end_h") is not None and r.get("wall_s") is not None:
            e = r["t_end_h"] * 3600
            iv.append((e - r["wall_s"], e))
    ev = sorted([(s, 1) for s, e in iv] + [(e, -1) for s, e in iv])
    cur = mx = 0; area = 0.0; last = ev[0][0]
    for t, d in ev:
        area += cur * (t - last); last = t
        cur += d; mx = max(mx, cur)
    span = ev[-1][0] - ev[0][0]
    print(f"{R.split('/')[-1]:12s} attempts {len(iv):4d}  max concurrent {mx:3d}  mean concurrent {area / span:5.1f}  span {span:5.0f}s")
