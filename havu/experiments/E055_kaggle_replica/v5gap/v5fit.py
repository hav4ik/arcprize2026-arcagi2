"""Which KV-per-context-token ratio (thrash 0.88:0.6) fits V5 58.19 and v3 50.97 once the per-pair pool has the corrected accuracy?"""
import statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts")
import v5proj_lib as P
for label, f in [("Kaggle-flag accuracy", (0.045, 0.054)), ("V5-measured accuracy", (0.097, 0.097))]:
    for ratio in (0.86, 0.89, 0.92):
        out = []
        for cfg, w, kv, s in (("V5", 36, 1.554e6, 1.04), ("v3", 48, 1.44e6, 1.0)):
            res = [P.run_pp(P.flipped(*f, sd), w, kv, ratio, (0.88, 0.6), s=s, reps=40) for sd in (11, 12)]
            out.append(f"{cfg} {statistics.mean(r[0] for r in res):5.1f}")
        print(f"{label:22s} ratio {ratio:.2f} thrash 0.88:0.6 -> " + " | ".join(out) + "   (LB: V5 58.19, v3 50.97)", flush=True)
