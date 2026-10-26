import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision")
from sim2 import *
M35 = multi_pool_files([("e128t_mt3_sub80_tx_b05", None)])
M4 = multi_pool_files([("e128t_mt3b_sub80_tx_b05", None)])
M7 = multi_pool_files([("e128t_final_all", None)])
MP = multi_pool_files([("e128t_mt3_sub80_tx_b05", None), ("e128t_mt3b_sub80_tx_b05", None), ("e128t_final_all", None)])
M6 = multi_pool_files([("e128t_mt6_sub80_tx_b025_ctx", None)])
TH = (0.88, 0.6)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
print("== reproduce: runs 3+5 records, thrash 0.88:0.6, s 1.04, ratio 0.83")
summ(run(dict(multi=M35, thrash=TH), reps), "base (runs3+5)")
summ(run(dict(multi=M35, thrash=None), reps), "base, no thrash")
print("== which multi records")
for lab, M in (("run4", M4), ("run7 (final harness)", M7), ("pooled 3,4,5,7", MP), ("run6 b0.25", M6)):
    summ(run(dict(multi=M, thrash=TH), reps), lab)
