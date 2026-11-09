import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision")
from sim2 import *
import json
MP = multi_pool_files([("e128t_mt3_sub80_tx_b05", None), ("e128t_mt3b_sub80_tx_b05", None), ("e128t_final_all", None)])
TH = (0.88, 0.6); reps = int(sys.argv[1])
S = [("S1 V5 ratio .72, pooled multi, gaps .04/.04", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04)),
     ("S2 V5 ratio .83 (doc), pooled, gaps .04/.04", dict(multi=MP, ratio5=0.83, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04)),
     ("S3 = S1 + multi acc -3pp (flip .045)", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, flip=0.045)),
     ("S4 = S1 + fast rerun s 1.15", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, s=1.15)),
     ("S5 = S2 + V5 L4 gap .07 (pods)", dict(multi=MP, ratio5=0.83, ratio6=0.83, thrash=TH, g5=0.07, g6=0.04)),
     ("S6 = S1 + slow rerun s 0.9", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, s=0.9)),
     ("S7 worst: S3 + s 1.15 + fpj 0.8 + tok 1.05", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, flip=0.045, s=1.15, fpj=0.8, tok_scale=1.05)),
     ("S8 tokens x1.05 only (S1)", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, tok_scale=1.05)),
     ("S9 tokens x1.10 only (S1)", dict(multi=MP, ratio5=0.72, ratio6=0.83, thrash=TH, g5=0.04, g6=0.04, tok_scale=1.10)),
     ]
res = {}
for lab, cfg in S:
    out = run(cfg, reps); summ(out, lab)
    res[lab] = [(o["v5"]["score"], o["v6"]["score"]) for o in out]
json.dump(res, open("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision/grid3.json", "w"))
