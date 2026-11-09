#!/usr/bin/env bash
# E070: Manh's image layouts on the DeepSeek API. E058's image arm (E046 best harness + --grid-format text+image
# --image-cell 0; baseline e058_pilot30) with only the layout of the training images changed:
#   pair  = --image-layout pair  (one image per training example, input left of output; Manh's pairs-plain)
#   sheet = --image-layout sheet (all training examples in one image, one per row; Manh's sheets)
# Test inputs stay one image each. Usage (repo root): [ATTEMPTS=2] [WORKERS=30] [BALANCE_STOP=40] run.sh <arm> [args...]
set -e
cd /home/vu/Workspace/ARC-AGI-2
ARM=$1; shift
case $ARM in
  pair|sheet) V=(--grid-format text+image --image-layout $ARM);;
  *) echo "usage: run.sh pair|sheet" >&2; exit 2;;
esac
TAG=e070_$ARM
export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  --model deepseek-flash --auto-peak-price --effort low --prompt lean2sub --answer-tool --final-mode session \
  --budget-notes 0.01 --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 \
  --image-cell 0 "${V[@]}" --attempts ${ATTEMPTS:-2} --tag $TAG --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} \
  --workers ${WORKERS:-30} --max-balance-stop ${BALANCE_STOP:-40} --allow-expensive-run "$@" > evals/results/log_$TAG.txt 2>&1
