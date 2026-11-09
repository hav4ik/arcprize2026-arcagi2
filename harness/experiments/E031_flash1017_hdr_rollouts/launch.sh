#!/usr/bin/env bash
# E031: DeepSeek-V4.1-Flash-1017 rollouts with s20x60 + --effort-header minimal (E030 pick) on the same 2,500
# pairs as the E028 SFT set (1,070 arc2_train + 1,430 nvarc_b0), 1 attempt per pair. --train-check is host-side only.
# Usage: launch.sh [arc2_train] [nvarc_b0]   Re-running resumes each output file.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
HERE=$ROOT/experiments/E031_flash1017_hdr_rollouts
E025=$ROOT/experiments/E025_train_rollouts/data
cd "$ROOT/evals"
CFG="--mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong \
  --effort-header minimal --turns-note 5 --train-check --price-multiplier 1 --auto-peak-price --attempts 1 \
  --workers ${WORKERS:-60} --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --max-balance-stop ${STOP:-40} \
  --allow-expensive-run"
for b in "${@:-arc2_train nvarc_b0}"; do
  for s in $b; do
    if [ "$s" = arc2_train ]; then
      ch=$ROOT/data/arc-agi_training_challenges.json; so=$ROOT/data/arc-agi_training_solutions.json
    else
      ch=$E025/${s}_challenges.json; so=$E025/${s}_solutions.json
    fi
    echo "=== $(date -u +%FT%TZ) start $s"
    ../.venv/bin/python run_deepseek_tools.py $CFG --challenges "$ch" --solutions "$so" \
      --pairs "$(cat "$HERE/pairs_$s.txt")" --tag "e031_$s" >> "results/log_e031_$s.txt" 2>&1
    echo "=== $(date -u +%FT%TZ) done $s"
  done
done
