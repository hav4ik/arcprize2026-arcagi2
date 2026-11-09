#!/usr/bin/env bash
# E025: DeepSeek Flash V3 (E017) rollouts, 1 attempt per pair, on the task pool built by build_dataset.py.
# Usage: launch.sh [batch ...]   (default: arc2_train nvarc_b0). Re-running resumes each output file.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
DATA=$ROOT/experiments/E025_train_rollouts/data
cd "$ROOT/evals"
V3="--mode grid --effort low --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 80000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 40000 --price-multiplier 1 \
  --auto-peak-price --train-check --prompt lean2roles --task-summary --helpers --attempts 1 \
  --workers ${WORKERS:-24} --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --max-balance-stop ${STOP:-40} \
  --allow-expensive-run"
for batch in "${@:-arc2_train nvarc_b0}"; do
  for b in $batch; do
    if [ "$b" = arc2_train ]; then
      ch=$ROOT/data/arc-agi_training_challenges.json; so=$ROOT/data/arc-agi_training_solutions.json
    else
      ch=$DATA/${b}_challenges.json; so=$DATA/${b}_solutions.json
    fi
    echo "=== $(date -u +%FT%TZ) start $b"
    ../.venv/bin/python run_deepseek_tools.py $V3 --challenges "$ch" --solutions "$so" --tag "e025_$b" \
      >> "results/log_e025_$b.txt" 2>&1
    echo "=== $(date -u +%FT%TZ) done $b"
  done
done
