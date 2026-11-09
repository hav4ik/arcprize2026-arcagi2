#!/usr/bin/env bash
# E061: Qwen3.8-Flash-Next (OpenRouter -> Alibaba) on DeepSeek's best tool harness (E046 `e046p_note_all`: sub80s40 +
# turns note 5 + fallback solve + multi-test with multi budget 0.5) plus E058's grid images (Qwen-aligned cells),
# 120 eval tasks x 2 attempts. DeepSeek's "Reasoning Effort: Minimal" header is left off (DeepSeek-specific).
# Compare with E058q (sub80s40 + images) and E048 (sub80s40, text). Resumes: only missing/errored attempts run.
#   run.sh [TAG] [WORKERS] [extra harness args, e.g. --tasks a,b]
TAG=${1:-e061q_best_img}; W=${2:-8}; shift 2 2>/dev/null
E=/home/vu/Workspace/ARC-AGI-2
cd $E/evals
CFG="$(cat $E/experiments/E048_qwen38_flash_openrouter/cfg_tools.txt)"
../.venv/bin/python run_deepseek_tools.py $CFG --turns-note 5 --fallback-solve --multi-test --multi-budget 0.5 \
  --grid-format text+image --image-cell 0 --image-align qwen \
  --sandbox-url http://172.18.0.2:6000 --attempts 2 --workers "$W" --tag "$TAG" "$@" >> results/log_$TAG.txt 2>&1
