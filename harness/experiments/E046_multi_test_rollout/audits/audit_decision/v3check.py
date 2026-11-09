import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision")
from sim2 import *
M35 = multi_pool_files([("e128t_mt3_sub80_tx_b05", None)])
for r in (0.72, 0.83):
    out = run(dict(multi=M35, w5=48, kv=1.437e6, ratio5=r, ratio6=r, s=1.0, thrash=(0.88, 0.6)), 40)
    v5 = [o["v5"] for o in out]
    print(f"v3 (per pair 48, KV 1.437M, s 1.0) ratio {r}: sim {statistics.mean(x['score'] for x in v5):.1f} (x0.96 = {0.96*statistics.mean(x['score'] for x in v5):.1f}), "
          f"a0miss {statistics.mean(x['a0miss'] for x in v5):.1f}, thrash h {statistics.mean(x['thr_h'] for x in v5):.1f}, first thrash {statistics.mean(x['first_thr'] for x in v5):.1f} h; LB 50.97")
