#!/usr/bin/env bash
# E038: finish the nvarc_b1 pool (1,246 tasks E036 did not run, incl. 1 errored) through OpenRouter -> DeepInfra,
# same s20x60 + header harness as E031/E036 (E037 showed parity). The guard stops new attempts below $STOP OpenRouter credit.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
HERE=$ROOT/experiments/E038_deepinfra_b1_finish
E025=$ROOT/experiments/E025_train_rollouts/data
cd "$ROOT/evals"
CFG="--mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong \
  --effort-header minimal --turns-note 5 --train-check --price-multiplier 1 --attempts 1 \
  --model deepseek/deepseek-v4.1-flash-1017 --openrouter-provider DeepInfra \
  --workers ${WORKERS:-64} --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --max-balance-stop ${STOP:-95} \
  --allow-expensive-run"
echo "=== $(date -u +%FT%TZ) start"
../.venv/bin/python run_deepseek_tools.py $CFG --challenges "$E025/nvarc_b1_challenges.json" --solutions "$E025/nvarc_b1_solutions.json" \
  --pairs "$(cat "$HERE/pairs_nvarc_b1r.txt")" --out "$ROOT/evals/results/e038_deepinfra_nvarc_b1r.jsonl" >> results/log_e038_nvarc_b1r.txt 2>&1
echo "=== $(date -u +%FT%TZ) done"
