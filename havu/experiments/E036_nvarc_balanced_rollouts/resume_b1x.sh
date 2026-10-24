#!/usr/bin/env bash
# After the main nvarc_b1 run (launch.sh pid given as $1) exits, resume the b1x top-up with the credit left.
# Skips if it is already 00:30 UTC or later (weekday peak pricing starts at 01:00).
set -euo pipefail
while kill -0 "$1" 2>/dev/null; do sleep 30; done
h=$(date -u +%H); m=$(date -u +%M)
if [ "$h" = 00 ] && [ "$m" -ge 30 ] || [ "$h" -ge 01 ] && [ "$h" -lt 12 ]; then echo "too late ($(date -u +%T)); not resuming"; exit 0; fi
ROOT=/home/vu/Workspace/ARC-AGI-2; cd $ROOT/evals
CFG=$(sed -n '/^CFG=/,/allow-expensive-run"/p' $ROOT/experiments/E036_nvarc_balanced_rollouts/launch.sh | tr -d '\\\n' | sed 's/^CFG="//; s/"$//; s/\${WORKERS:-60}/64/; s/\${SANDBOX:-http:\/\/172.18.0.2:6000}/http:\/\/172.18.0.2:6000/; s/\${STOP:-40}/4/')
echo "resuming b1x at $(date -u +%T)"
../.venv/bin/python run_deepseek_tools.py $CFG --challenges ../experiments/E025_train_rollouts/data/nvarc_b1_challenges.json \
  --solutions ../experiments/E025_train_rollouts/data/nvarc_b1_solutions.json \
  --pairs "$(cat ../experiments/E036_nvarc_balanced_rollouts/pairs_nvarc_b1x.txt)" --tag e036_nvarc_b1x >> results/log_e036_nvarc_b1x.txt 2>&1
echo "b1x done at $(date -u +%T)"
