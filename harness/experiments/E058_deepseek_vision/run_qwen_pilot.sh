#!/usr/bin/env bash
# E058 Qwen arm: the E048 tool harness (sub80s40, Qwen3.8 Flash via OpenRouter -> Alibaba) plus text+image with
# Qwen-aligned cells (32*k px), on the same 30 pilot tasks; baseline = E048 text-only on those tasks.
cd /home/vu/Workspace/ARC-AGI-2/evals
CFG="$(cat ../experiments/E048_qwen38_flash_openrouter/cfg_tools.txt)"
../.venv/bin/python run_deepseek_tools.py $CFG --grid-format text+image --image-cell 0 --image-align qwen \
  --tasks "$(cat ../experiments/E058_deepseek_vision/pilot_tasks.txt)" --attempts 2 --workers 8 --sandbox-url http://172.18.0.2:6000 \
  --tag e058q_pilot30 > results/log_e058q_pilot30.txt 2>&1
