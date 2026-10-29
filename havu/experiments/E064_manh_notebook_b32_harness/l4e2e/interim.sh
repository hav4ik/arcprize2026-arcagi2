#!/bin/bash
# E064b: mirror the B32-M replica on L4 pod #2 (read-only rsync, no scratch) and write an interim report:
#   reports/b32m_x2t_<tag>.md (replica_report.py) + reports/interim_<tag>.txt (throughput and attempt health vs pod 1's
#   manh_x2t_w80 = same fork config on our build / Python 3.12, and candBf_x2t = B32 on the old pod, at the same pod hour).
set -euo pipefail
cd /home/vu/Workspace/ARC-AGI-2/experiments/E064_manh_notebook_b32_harness/l4e2e
TAG=${1:-$(date -u +%H%M)}; RUN=b32m_x2t
B=/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit
M=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4
rsync -a --exclude scratch -e "ssh -p 11660 -o ConnectTimeout=20 -o BatchMode=yes" root@213.173.105.27:/root/runs/$RUN/ runs/$RUN/
/home/vu/Workspace/ARC-AGI-2/.venv/bin/python $B/replica_report.py runs/$RUN --out reports/${RUN}_$TAG.md > /dev/null
H=$(python3 -c "import json; r=[json.loads(l) for l in open('runs/$RUN/out/dsv41_logs/metrics.jsonl')]; print(r[-1]['elapsed_h'])")
{ echo "# interim $TAG UTC, $RUN at pod $H h"; echo
  python3 $M/compare_tps.py runs/$RUN=b32m $M/runs/manh_x2t_w80=manh80_pod1 $B/runs/candBf_x2t=candBf_x2t_oldpod --upto $H
  echo; python3 $M/attempt_stats.py runs/$RUN=b32m $M/runs/manh_x2t_w80=manh80_pod1 $B/runs/candBf_x2t=candBf_x2t --upto $H
  echo; grep -h -E "ERROR|Traceback|OutOfMemory|CUDA out of memory|server died|restart" runs/$RUN/out/dsv41_logs/solver.log runs/$RUN/out/dsv41_logs/server.log 2>/dev/null | tail -5
  echo; tail -1 runs/$RUN/cgmem.log 2>/dev/null; tail -1 runs/$RUN/cpu.log 2>/dev/null; tail -1 runs/$RUN/mem.log 2>/dev/null
  echo; grep -E "Kaggle score|On Kaggle" reports/${RUN}_$TAG.md | head -3
} > reports/interim_$TAG.txt
cat reports/interim_$TAG.txt
