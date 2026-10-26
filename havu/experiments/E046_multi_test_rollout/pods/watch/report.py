"""One row per current run from hist_<run>.txt: age, tok/s, hits, uncached, KV, running, holds, attempts, max KV, min/max running."""
import re, sys
pat = re.compile(r"\[(\S+) \+([\d.]+)h\] server: ([\d.]+) gen tok/s, prefix hits ([\d.]+), uncached ([\d.]+) tok/s, KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?; (\d+) attempts done")
print("| run | age | tok/s | hits | uncached | KV | running | holds | attempts | max KV (at) | running min-max (>=0.3h) | restarts |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in sys.argv[1:]:
    L = open(f"hist_{run}.txt").read().splitlines()
    W = [m.groups() for m in map(pat.search, L) if m]
    if not W:
        print(f"| {run} | (no server window yet) |"); continue
    t, h, tps, hits, unc, kv, rn, wt, pre, att = W[-1]
    mk = max(W, key=lambda w: float(w[5]))
    rr = [float(w[6]) for w in W if float(w[1]) >= 0.3] or [float(rn)]
    g = sum("KV gate" in l for l in L); ns = sum("server start" in l for l in L)
    print(f"| {run} | {float(h):.2f}h | {tps} | {hits} | {unc} | {kv} | {float(rn):.0f} | {g} | {att} | {mk[5]} ({float(mk[1]):.2f}h) | {min(rr):.0f}-{max(rr):.0f} | {ns - 1} |")
