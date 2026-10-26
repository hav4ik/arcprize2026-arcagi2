#!/bin/bash
# E048: attempt 0 with an optional second guess (--alt-answer), all 120 eval tasks, Flash-1017 via OpenRouter, the
# Kaggle candidate's flags (as E046 e046p_note_all). Each attempt index is a fresh sample.
# Usage (repo root): [PROVIDER=DeepInfra] [ATTEMPTS=2] run.sh <passive|active>
set -e
M=$1; R=evals/results; OUT=$R/tools_persistent_grid_lean2sub_deepseek-flash_low_e048or_$M.jsonl
.venv/bin/python evals/run_deepseek_tools.py --challenges data/arc-agi-2-github/data/evaluation \
  --model deepseek/deepseek-v4.1-flash-1017 --openrouter-provider ${PROVIDER:-DeepInfra} --effort low --prompt lean2sub \
  --answer-tool --final-mode session --budget-notes 0.01 --budget-warn 0.75 --lenient-parse --budget-soft 40000 \
  --effort-header minimal --max-tokens 64000 --max-total-output 80000 --max-turns 40 --attempts ${ATTEMPTS:-2} \
  --multi-test --multi-budget 0.5 --fallback-solve --train-check --turns-note 5 --alt-answer $M \
  --tag e048or_$M --out $OUT --sandbox-url http://172.18.0.2:6000 --workers ${WORKERS:-48} \
  --max-balance-stop ${BALANCE_STOP:-20} --allow-expensive-run > $R/log_e048or_$M.txt 2>&1
