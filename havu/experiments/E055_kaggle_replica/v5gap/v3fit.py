import statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts")
import v5proj_lib as P
for label, f in [("w10_u2 accuracy (as projected)", (0.0, 0.0)), ("Kaggle-flag accuracy", (0.045, 0.054)), ("V5-measured accuracy", (0.097, 0.097))]:
    for ratio, th in ((0.72, (0.88, 0.6)), (0.78, (0.88, 0.6)), (0.83, (0.88, 0.6))):
        seeds = (0,) if f == (0.0, 0.0) else (11, 12)
        res = [P.run_pp(P.flipped(*f, sd), 48, 1.44e6, ratio, th, s=1.0, reps=40) for sd in seeds]
        print(f"v3 x48 KV 1.44M  {label:30s} ratio {ratio:.2f} thrash {th}: {statistics.mean(r[0] for r in res):5.1f}  (LB 50.97)", flush=True)
