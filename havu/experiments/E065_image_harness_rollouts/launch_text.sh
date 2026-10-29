#!/usr/bin/env bash
# E065 text twin: the same pairs with the E046 best harness WITHOUT images (the Kaggle-tested harness), so pruning /
# SFT can target either harness. Otherwise identical to launch.sh: DeepSeek API, 1 attempt per pair, off-peak only (no attempt starts in a weekday peak window or within 20 min of
# one). Same pairs as E031 (arc2_train, nvarc_b0) + E036/E038 (all 7,500 nvarc_b1, parent-balanced build order).
# Usage (repo root): launch.sh BATCH [PAIRS_FILE]   BATCH = arc2_train | nvarc_b0 | nvarc_b1; re-running resumes
#   (the output file is per batch; PAIRS_FILE = a subset, e.g. pairs_nvarc_b0_pilot.txt).
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E065_image_harness_rollouts
E025=$ROOT/experiments/E025_train_rollouts/data
cd "$ROOT"
export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
B=$1; PF=${2:-$HERE/pairs_$B.txt}
if [ "$B" = arc2_train ]; then ch=$ROOT/data/arc-agi_training_challenges.json; so=$ROOT/data/arc-agi_training_solutions.json
else ch=$E025/${B}_challenges.json; so=$E025/${B}_solutions.json; fi
echo "=== $(date -u +%FT%TZ) start $B ($(tr ',' '\n' < "$PF" | grep -c .) pairs)"
.venv/bin/python evals/run_deepseek_tools.py --challenges "$ch" --solutions "$so" --pairs "$(cat "$PF")" \
  --model deepseek-flash --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 \
  --attempts 1 --auto-peak-price --offpeak-only 20 --tag e065t_$B \
  --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --workers ${WORKERS:-32} --max-balance-stop ${STOP:-5} \
  --allow-expensive-run >> evals/results/log_e065t_$B.txt 2>&1
echo "=== $(date -u +%FT%TZ) done $B"
