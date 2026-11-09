import re, sys, os
pat = re.compile(r"\+([\d.]+)h\] server: .*KV ([\d.]+), running")
for run in sys.argv[1:]:
    f = f"old/hist_{run}.txt" if os.path.exists(f"old/hist_{run}.txt") else f"hist_{run}.txt"
    W = [(float(a), float(b)) for a, b in (m.groups() for m in map(pat.search, open(f)) if m)]
    out = []
    first = None
    for i, (h, kv) in enumerate(W):
        win = [(x, y) for x, y in W if h - 0.5 <= x <= h]
        if h < 0.8 or len(win) < 5:
            continue
        n = len(win); mx = sum(x for x, _ in win) / n; my = sum(y for _, y in win) / n
        s = sum((x - mx) * (y - my) for x, y in win) / sum((x - mx) ** 2 for x, _ in win)
        out.append(f"{h:.2f}h {s:+.2f}")
        if s < 0.05 and first is None:
            first = (h, kv)
    print(f"{run}: 0.5h-window KV slope/h: " + ", ".join(out[::2]) + f" | first slope<0.05: {first}")
