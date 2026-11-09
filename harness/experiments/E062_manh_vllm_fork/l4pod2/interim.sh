#!/bin/bash
# E062 L4 pod #2: mirror the live replica (read-only) and write an interim report, plus a paired report against
# candBf_x2t (B32 on the same workload) cut at the same solver hour.
#   bash interim.sh [NAME]      (default b32r1_x2t)
set -euo pipefail
NAME=${1:-b32r1_x2t}
cd /home/vu/Workspace/ARC-AGI-2
E=experiments/E062_manh_vllm_fork/l4pod2; T=experiments/E055_kaggle_replica/toolkit
rsync -a --exclude scratch -e "ssh -o BatchMode=yes -p 11660" root@213.173.105.27:/root/runs/$NAME/ $E/runs/$NAME/
H=$(python3 -I -c "
import re,sys
h=0.0
for l in open('$E/runs/$NAME/out/dsv41_logs/solver.log'):
    m=re.match(r'^\[\S+ \+([0-9.]+)h\]', l)
    if m: h=float(m.group(1))
print(f'{h:.2f}')")
.venv/bin/python $T/replica_report.py $E/runs/$NAME --no-traces --out $E/runs/$NAME/report_live.md > /dev/null
C=$E/runs/cut_candBf_x2t_${H}h
rm -rf $C; python3 -I $E/cut_run.py $T/runs/candBf_x2t $H $C > /dev/null
.venv/bin/python $T/replica_report.py $C --no-traces --out $C/report.md > /dev/null
.venv/bin/python $T/replica_report.py $E/runs/$NAME --no-traces --baseline $C --out $E/runs/$NAME/report_vs_candBf_${H}h.md > /dev/null
echo "solver hour $H: $E/runs/$NAME/report_live.md, $E/runs/$NAME/report_vs_candBf_${H}h.md"
grep -m2 -E "Kaggle score \(task-weighted\)|paired" $E/runs/$NAME/report_vs_candBf_${H}h.md | cut -c1-300
grep -m1 -E "Kaggle score \(task-weighted\)" $C/report.md | cut -c1-200
