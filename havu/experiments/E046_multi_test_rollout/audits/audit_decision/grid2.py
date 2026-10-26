import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision")
from sim2 import *
import json
M35 = multi_pool_files([("e128t_mt3_sub80_tx_b05", None)])
M4 = multi_pool_files([("e128t_mt3b_sub80_tx_b05", None)])
MP = multi_pool_files([("e128t_mt3_sub80_tx_b05", None), ("e128t_mt3b_sub80_tx_b05", None), ("e128t_final_all", None)])
reps = int(sys.argv[1]); part = sys.argv[2]
TH = (0.88, 0.6)
S = []
if part == "a":  # KV ratio calibration x thrash, runs 3+5 records (the doc's)
    for r5, r6, lab in ((0.83, 0.83, "doc ratio 0.83/0.83"), (0.72, 0.83, "pod-calibrated 0.72(V5)/0.83(V6)"), (0.72, 0.72, "0.72/0.72"), (0.95, 0.95, "0.95/0.95")):
        for th in (None, TH, (0.85, 0.5), (0.82, 0.6)):
            S.append((f"{lab} thrash {th}", dict(multi=M35, ratio5=r5, ratio6=r6, thrash=th)))
if part == "b":  # pessimistic V6 knobs at pod-calibrated ratios, thrash 0.88:0.6
    base = dict(multi=M35, ratio5=0.72, ratio6=0.83, thrash=TH)
    S.append(("cal base", base))
    S.append(("cal, V6 multi tokens x1.15", dict(base, tok_scale=1.15)))
    S.append(("cal, V6 multi first prompt 1.0x sum", dict(base, fpj=1.0)))
    S.append(("cal, V6 flip 0.045 (-3pp multi acc)", dict(base, flip=0.045)))
    S.append(("cal, V6 flip 0.09 (-6pp multi acc)", dict(base, flip=0.09)))
    S.append(("cal, run4 records (worst b0.5 run)", dict(base, multi=M4)))
    S.append(("cal, pooled 3,4,5,7", dict(base, multi=MP)))
    S.append(("cal, s 0.85 (slower rerun)", dict(base, s=0.85)))
    S.append(("cal, s 1.15", dict(base, s=1.15)))
    S.append(("cal, L4 gap g5 .04 g6 .06", dict(base, g5=0.04, g6=0.06)))
    S.append(("cal, no thrash, flip .045, tok 1.15, fpj 1.0", dict(base, thrash=None, flip=0.045, tok_scale=1.15, fpj=1.0)))
    S.append(("cal, no thrash, flip .09, g6 .02 extra", dict(base, thrash=None, flip=0.09, g6=0.02)))
    S.append(("cal, s .85, flip .045, tok 1.15", dict(base, s=0.85, flip=0.045, tok_scale=1.15)))
if part == "c":  # alternatives (V6 slot varies; V5 fixed at per pair 36 ungated)
    base = dict(multi=M35, ratio5=0.72, ratio6=0.83, thrash=TH)
    for w6, g6 in ((36, 0.6), (32, 0), (36, 0), (36, 0.7), (40, 0.6), (44, 0.6), (44, 0.7), (32, 0.6), (48, 0.6)):
        S.append((f"cal, V6 alt W{w6} gate {g6}", dict(base, w6=w6, gate6=g6)))
    for w6, g6 in ((36, 0.6), (32, 0), (36, 0), (44, 0.6)):
        S.append((f"cal 90% KV pool, V6 alt W{w6} gate {g6}", dict(base, w6=w6, gate6=g6, kv=1.403e6)))
res = {}
for lab, cfg in S:
    d = summ(run(cfg, reps), lab)
    res[lab] = d
json.dump(res, open(f"/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision/grid2_{part}.json", "w"))
