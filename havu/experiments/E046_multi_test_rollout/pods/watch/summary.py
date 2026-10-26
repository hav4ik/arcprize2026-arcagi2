"""Per-run KV/throughput summary from hist_<run>.txt (+ mem_<run>.log). Usage: summary.py <dir> run [run ...]"""
import re, statistics, sys, os
D = sys.argv[1]
pat = re.compile(r"\[(\S+) \+([\d.]+)h\] server: ([\d.]+) gen tok/s, prefix hits ([\d.]+), uncached ([\d.]+) tok/s, KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?; (\d+) attempts done")
for run in sys.argv[2:]:
    f = f"{D}/old/hist_{run}.txt"
    if not os.path.exists(f):
        f = f"{D}/hist_{run}.txt"
    L = open(f).read().splitlines()
    W = []
    for l in L:
        m = pat.search(l)
        if m:
            t, h, tps, hits, unc, kv, rn, wt, pre, att = m.groups()
            W.append(dict(t=t, h=float(h), tps=float(tps), hits=float(hits), unc=float(unc), kv=float(kv), rn=float(rn), pre=float(pre), att=int(att)))
    gates = [l for l in L if "KV gate" in l]
    up = [l for l in L if "server up" in l]
    starts = [l for l in L if "server start" in l]
    steady = [w for w in W if w["h"] >= 0.3]
    plateau = statistics.median([w["tps"] for w in steady if w["kv"] < 0.75]) if [w for w in steady if w["kv"] < 0.75] else None
    thr = [w for w in steady if w["h"] >= 0.4 and ((plateau and w["tps"] < 0.75 * plateau and w["kv"] > 0.8) or (w["unc"] > 350 and w["hits"] < 0.9))]
    mem = f"{D}/old/mem_{run}.log" if os.path.exists(f"{D}/old/mem_{run}.log") else f"{D}/mem_{run}.log"
    mx = None
    if os.path.exists(mem):
        vals = [float(re.search(r"rss_gib=([\d.]+)", l).group(1)) for l in open(mem) if "rss_gib=" in l]
        late = vals[3:] or vals
        mx = (max(vals) if vals else None, max(late) if late else None)
    last = W[-1] if W else None
    print(f"## {run}: {len(starts)} server start(s); {up[0].split(';')[1].strip() if up else 'no server up'}")
    if last:
        print(f"  latest +{last['h']:.2f}h ({last['t']}): {last['tps']} tok/s, hits {last['hits']}, uncached {last['unc']}, KV {last['kv']}, running {last['rn']:.0f}, {last['att']} attempts; preemptions max {max(w['pre'] for w in W)}")
        print(f"  plateau (median, >=0.3h, KV<0.75): {plateau}; max KV {max(w['kv'] for w in W)} at +{max(W, key=lambda w: w['kv'])['h']:.2f}h")
        # KV/tok/s every ~0.25 h
        pts, nxt = [], 0.25
        for w in W:
            if w["h"] >= nxt:
                pts.append(f"{w['h']:.2f}h {w['tps']:.0f}/{w['kv']:.2f}")
                nxt = w["h"] + 0.24
        print("  tok/s/KV: " + ", ".join(pts))
    print(f"  thrash windows: {len(thr)}" + ("; first +%.2fh; " % thr[0]["h"] + "; ".join(f"+{w['h']:.2f}h {w['tps']:.0f} tok/s h{w['hits']} u{w['unc']:.0f} KV{w['kv']}" for w in thr) if thr else ""))
    print(f"  gate holds: {len(gates)}" + (f"; first {gates[0][:60]}; last {gates[-1][:90]}" if gates else ""))
    print(f"  mem.log rss_gib max {mx[0] if mx else None} (excluding the first 3 min: {mx[1] if mx else None})")
