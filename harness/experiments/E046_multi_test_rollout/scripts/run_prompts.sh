#!/bin/bash
# E046 prompt formats (--multi-prompt) on the DeepSeek API, Flash-1017, sub80s40 + header + the Kaggle candidate's
# multi options (--fallback-solve --multi-budget 0.5 --train-check --turns-note 5). Usage (repo root): run_prompts.sh <variant> [all]
# "all" runs every eval task (single-test tasks too), else the 45 multi-test tasks. Output tag e046p_<variant>[_all].
set -e
V=$1; SCOPE=${2:-multi}
export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
TASKS=(); TAG=e046p_$V
if [ "$SCOPE" != all ]; then TASKS=(--tasks "$(paste -sd, experiments/E046_multi_test_rollout/multi_tasks.txt)"); else TAG=${TAG}_all; fi
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  --model deepseek-flash --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --attempts 2 "${TASKS[@]}" --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 \
  --multi-prompt "$V" --auto-peak-price --tag $TAG --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} \
  --workers ${WORKERS:-45} --max-balance-stop ${BALANCE_STOP:-15} ${ALLOW:+--allow-expensive-run} > evals/results/log_$TAG.txt 2>&1
