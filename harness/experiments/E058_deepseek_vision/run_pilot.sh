#!/usr/bin/env bash
# E058: E046's best harness (sub80s40 + minimal header + multi-test + multi-budget 0.5 + fallback-solve) plus
# --grid-format text+image --image-cell 0 (JSON prompt + one DeepSeek-aligned image per grid).
# Usage: run_pilot.sh TAG [extra harness args...]
set -e
cd /home/vu/Workspace/ARC-AGI-2
export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
TAG=$1; shift
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  --model deepseek-flash --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 \
  --grid-format text+image --image-cell 0 --auto-peak-price --tag $TAG --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} \
  --workers ${WORKERS:-30} --max-balance-stop ${BALANCE_STOP:-15} "$@" > evals/results/log_$TAG.txt 2>&1
