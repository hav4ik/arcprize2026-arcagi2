import json, random, statistics as st
S = json.load(open('/home/vu/Workspace/ARC-AGI-2/experiments/E042_dsv41_l4_synthetic/metadata/trace_shapes/tools_e030_hdrmin.json'))
CAP = 30000
def capped(a):
    out, g = [], 0
    for c, r in a["turns"]:
        if g + c >= CAP:
            out.append((CAP - g, 0)); g = CAP; break
        out.append((c, r)); g += c
    return out, g
caps = [capped(a) for a in S]
gens = [g for _, g in caps]; raw = [sum(c for c, _ in a["turns"]) for a in S]
print(f"attempts {len(S)}: raw gen mean {st.mean(raw):,.0f} p50 {sorted(raw)[len(raw)//2]:,} p90 {sorted(raw)[int(.9*len(raw))]:,}; "
      f"hit 30K cap: {sum(x >= CAP for x in raw)/len(raw):.0%}; capped mean {st.mean(gens):,.0f}, turns mean {st.mean(len(t) for t,_ in caps):.1f}")

def sim(C, G, label, T=12*3600, setup=720, cold_tok=7200, prefill=1190, tool_s=2.0, policy="adaptive", seed=0):
    """Slots run capped attempts back to back. Aggregate generated tok/s = G(active); per stream G/active.
    Cold start: the first prompts of all C slots prefill at `prefill` tok/s before decode reaches full speed."""
    rng = random.Random(seed)
    dt = 5.0
    t = setup + C * cold_tok / prefill / 2   # half of the cold prefill overlaps decode
    slots = []
    def new():
        turns, g = rng.choice(caps)
        return {"turns": list(turns), "left": turns[0][0], "wait": 0.0, "start": t, "gen": g}
    for _ in range(C): slots.append(new())
    done, durs, cut = 0, [], 0
    while t < T and any(slots):
        act = [s for s in slots if s and s["wait"] <= 0]
        n = len(act)
        rate = G(n) / n if n else 0
        for s in slots:
            if not s: continue
            if s["wait"] > 0: s["wait"] -= dt; continue
            s["left"] -= rate * dt
            if s["left"] <= 0:
                s["turns"].pop(0)
                if s["turns"]:
                    s["left"] = s["turns"][0][0]; s["wait"] = tool_s
                else:
                    done += 1; durs.append(t - s["start"]); s.clear()
        t += dt
        for i, s in enumerate(slots):
            if not s:
                left = T - t
                n_now = max(1, sum(1 for x in slots if x))
                ok = (left * G(n_now + 1) / (n_now + 1) * 0.85 >= CAP) if policy == "adaptive" else (left >= CAP / (G(C) / C))
                slots[i] = new() if ok else None
    cut = sum(1 for s in slots if s)
    d = sorted(durs)
    print(f"{label}: {done} rollouts in 12 h ({done/167:.1f} per pair for 167 pairs), unfinished at deadline {cut}, "
          f"attempt wall p50 {d[len(d)//2]/3600:.2f} h p90 {d[int(.9*len(d))]/3600:.2f} h max {d[-1]/3600:.2f} h")
    return done

# Aggregate generated tok/s in tool mode vs active streams. E48: run 7 settled 505 at 128; lower C scaled from Kaggle
# decode (K5 32K x 128 736, 32K x 64 552; K4 8K x 32 326, 32K x 8 146) by the same 505/736 tool-mode factor.
import bisect
def interp(pts):
    xs, ys = zip(*pts)
    def G(n):
        if n <= xs[0]: return ys[0] * n / xs[0]
        if n >= xs[-1]: return ys[-1]
        i = bisect.bisect(xs, n); x0, x1, y0, y1 = xs[i-1], xs[i], ys[i-1], ys[i]
        return y0 + (y1 - y0) * (n - x0) / (x1 - x0)
    return G
f = 505 / 736
E48 = interp([(1, 35 * f), (8, 146 * f), (32, 326 * f), (64, 552 * f), (128, 505)])
for pol in ("adaptive", "fixed"):
    sim(128, E48, f"E48 x 128 ({pol} launch cutoff)", policy=pol)
for lo in (470,):
    G = interp([(1, 35 * f * lo / 505), (8, 146 * f * lo / 505), (32, 326 * f * lo / 505), (64, 552 * f * lo / 505), (128, lo)])
    sim(128, G, f"E48 x 128, steady {lo} tok/s")

print("--- E64 x 64, settled tool-mode tok/s assumed (run 8 pending)")
for g64 in (400, 450):
    k = g64 / 510   # E64 Kaggle decode 1K x 64 with PR #154: 510; 8K x 32: 326; 32K x 8: 146
    G = interp([(1, 35 * k), (8, 146 * k), (32, 326 * k), (64, g64)])
    sim(64, G, f"E64 x 64 at {g64}")
print("--- no cap (full E030 lengths), E48 x 128")
CAP = 10**9
caps[:] = [capped(a) for a in S]
print("max gen", max(g for _, g in caps))
sim(128, E48, "E48 x 128 uncapped, launch until the end", policy="fixed")
