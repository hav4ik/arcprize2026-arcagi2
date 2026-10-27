import statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts")
import v5proj_lib as P
for label, f in [("Kaggle-flag accuracy", (0.045, 0.054)), ("V5-measured accuracy", (0.097, 0.097))]:
    for ratio in (0.74, 0.78):
        res = [P.run_pp(P.flipped(*f, sd), 36, 1.554e6, ratio, (0.88, 0.6), s=1.04, reps=40) for sd in (11, 12)]
        print(f"V5 {label:22s} ratio {ratio} thrash 0.88:0.6: {statistics.mean(r[0] for r in res):.1f}", flush=True)
