#!/bin/bash
# V12-fast vs V12 at the same Kaggle clock: sync, hourly score/tps/running/cap-reserved, serving metrics. usage: report.sh
P=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h; D=$P/v12_fast/data; RUN=v12fast2_rerun
SOL=/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json
V=$P/manh_v12/data; R=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/sftv01/v12ref/replicas/manh_v12; VS=$R/samp.jsonl
REPLICA_RUN=$RUN REPLICA_DATA=$D bash $P/v12_fast/scripts/sync.sh > /dev/null 2>&1
T0=$(ssh -p 29668 root@157.157.221.29 "python3 -c \"import os; print(os.path.getmtime('/root/nbruns/$RUN/notebook.sha256'))\"")
NOW_H=$(python3 -c "import time; print(round((time.time() - $T0) / 3600, 2))")
echo "Kaggle h now $NOW_H"
python3 $P/scripts/hourly_vs_v12.py $SOL $D/dsv41_logs/attempts.jsonl $D/tps.tsv $D/nbruns/$RUN/samp.jsonl $T0 $R/attempts.jsonl $V/tps.tsv $VS 1793325064.3 $NOW_H
python3 $P/scripts/hourly_extra_vs_v12.py $SOL $D/dsv41_logs/metrics.jsonl $D/dsv41_logs/attempts.jsonl $D/nbruns/$RUN/samp.jsonl $T0 $V/dsv41_logs/metrics.jsonl $R/attempts.jsonl $VS 1793325064.3 $NOW_H
