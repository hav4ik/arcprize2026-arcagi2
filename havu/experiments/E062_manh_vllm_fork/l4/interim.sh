#!/bin/bash
# E062 L4: mirror the Manh replica (read-only rsync, no scratch) and write the interim reports:
#   reports/manh_x2t_w80_<tag>.md (replica_report.py), plus the throughput and attempt-health tables vs B32
#   (this pod's b32pod_x2t for hour 0-1, the old pod's candBf_x2t for all hours) in reports/interim_<tag>.txt
set -euo pipefail
cd /home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4
TAG=${1:-$(date -u +%H%M)}; RUN=${RUN:-manh_x2t_w80}
B=/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit
rsync -a --exclude scratch -e "ssh -p 19843 -o ConnectTimeout=20" root@213.173.105.17:/root/runs/$RUN/ runs/$RUN/
/home/vu/Workspace/ARC-AGI-2/.venv/bin/python $B/replica_report.py runs/$RUN --out reports/${RUN}_$TAG.md > /dev/null
H=$(python3 -c "import json; r=[json.loads(l) for l in open('runs/$RUN/out/dsv41_logs/metrics.jsonl')]; print(r[-1]['elapsed_h'])")
{ echo "# interim $TAG UTC, $RUN at pod $H h"; echo
  python3 compare_tps.py runs/$RUN=manh80 runs/b32pod_x2t=b32pod_samepod $B/runs/candBf_x2t=candBf_x2t_oldpod --upto $H
  echo; python3 attempt_stats.py runs/$RUN=manh80 $B/runs/candBf_x2t=candBf_x2t --upto $H
  echo; tail -3 runs/$RUN/out/dsv41_logs/solver.log | cut -c1-250
  echo; tail -1 runs/$RUN/cgmem.log 2>/dev/null; tail -1 runs/$RUN/cpu.log
  echo; grep -E "Kaggle score|central" reports/${RUN}_$TAG.md | head -4
} > reports/interim_$TAG.txt
cat reports/interim_$TAG.txt
