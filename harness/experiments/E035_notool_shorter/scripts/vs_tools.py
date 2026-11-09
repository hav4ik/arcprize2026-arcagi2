"""No-tool arms vs the cheapest tool config in the noise band (E030 s20x60 + header): paired task-level bootstrap."""
import random, statistics as st
from abs_table import RUNS, TASKS, load, tok, score, complete
REF = "E030 s20x60 + header"
files = {l: f for l, f, _ in RUNS}

def cost(recs):
    tot = 0.0
    for r in recs.values():
        if "est_cost_usd" in r:
            tot += r["est_cost_usd"]
        else:
            u = r["usage"]
            tot += (u["completion_tokens"] * 0.60 + u["prompt_tokens"] * 0.14) / 1e6
    return tot

def per_task(recs, t):
    n = TASKS[t]
    p2 = sum(any(recs.get((t, p, a), {}).get("correct") for a in (0, 1)) for p in range(n)) / n
    p1 = sum(bool(recs.get((t, p, a), {}).get("correct")) for p in range(n) for a in (0, 1)) / (2 * n)
    return p2, p1

ref = load(files[REF])
print("| Arm | Tasks | pass@2 arm / tools | Δ pass@2 [95% CI] | pass@1 arm / tools | Δ pass@1 [95% CI] | tokens mean arm / tools | $ per call arm / tools |")
print("|---|---|---|---|---|---|---|---|")
random.seed(0)
for label in ["E034 max", "E035 D low + header + stats", "E035 C low + stats", "E035 E low + header",
              "E035 A max + stats (partial)", "E035 B low (partial)"]:
    recs = load(files[label])
    ts = complete(recs)
    a = {t: per_task(recs, t) for t in ts}; b = {t: per_task(ref, t) for t in ts}
    def ci(i):
        d = [a[t][i] - b[t][i] for t in ts]
        bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
        return 100 * sum(d) / len(d), 100 * bs[125], 100 * bs[4875]
    A2, A1 = score(recs, ts); B2, B1 = score(ref, ts)
    d2, d1 = ci(0), ci(1)
    sub = {k: v for k, v in ref.items() if k[0] in ts}
    ra = {k: v for k, v in recs.items() if k[0] in ts}
    print(f"| {label} | {len(ts)} | {A2:.1f}% / {B2:.1f}% | {d2[0]:+.1f} [{d2[1]:+.1f}, {d2[2]:+.1f}] | {A1:.1f}% / {B1:.1f}% | "
          f"{d1[0]:+.1f} [{d1[1]:+.1f}, {d1[2]:+.1f}] | {st.mean(tok(r) for r in ra.values())/1e3:.1f}k / "
          f"{st.mean(tok(r) for r in sub.values())/1e3:.1f}k | {1000*cost(ra)/len(ra):.1f}m / {1000*cost(sub)/len(sub):.1f}m |")
