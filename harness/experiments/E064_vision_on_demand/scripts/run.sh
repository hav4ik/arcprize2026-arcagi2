#!/usr/bin/env bash
# E064: E058's DeepSeek image arm (E046 best harness + --grid-format text+image --image-cell 0) with one vision change:
#   show   = + --show-tool (model-chosen images of any session grid, in tool results)
#   check  = + --check-image (the first accepted answer is shown back as images, then resubmitted or fixed)
#   strict = + --image-note strict (confirm what the images suggest in code)
#   lazy   = JSON only (no up-front images) + --show-tool
# Usage (repo root): [VIA=or] [ATTEMPTS=2] [WORKERS=30] [BALANCE_STOP=3] run.sh <arm> [extra harness args...]
#   VIA unset = DeepSeek API (tag e064_<arm>); VIA=or = OpenRouter, provider ${PROVIDER:-DeepInfra} (tag e064or_<arm>),
#   with --image-in user (DeepInfra rejects image parts in tool results). OpenRouter's DeepSeek provider is blocked by
#   the account's training-data setting. Arm base = E058 replicate (text+image only), the baseline for VIA=or arms.
set -e
cd /home/vu/Workspace/ARC-AGI-2
ARM=$1; shift
case $ARM in
  show) V=(--grid-format text+image --show-tool);;
  check) V=(--grid-format text+image --check-image);;
  strict) V=(--grid-format text+image --image-note strict);;
  lazy) V=(--show-tool);;
  *) V=(--grid-format text+image);;  # base: E058 replicate
esac
if [ "$VIA" = or ]; then
  TAG=e064or_$ARM; M=(--model deepseek/deepseek-v4.1-flash-1017 --openrouter-provider ${PROVIDER:-DeepInfra} --image-in user)
  OUT=(--out evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_$TAG.jsonl)
else
  TAG=e064_$ARM; M=(--model deepseek-flash --auto-peak-price); OUT=()
  export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
fi
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  "${M[@]}" --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 \
  --image-cell 0 "${V[@]}" --attempts ${ATTEMPTS:-2} --tag $TAG "${OUT[@]}" --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} \
  --workers ${WORKERS:-30} --max-balance-stop ${BALANCE_STOP:-3} --allow-expensive-run "$@" > evals/results/log_$TAG.txt 2>&1
