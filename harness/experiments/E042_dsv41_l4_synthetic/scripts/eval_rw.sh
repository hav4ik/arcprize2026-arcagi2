#!/usr/bin/env bash
# E042 RW: E043's tool ARC eval (eval_arc.sh with PRESET=sub80s40, 2 attempts) against the real tool E128 2-bit
# checkpoint served on the Runpod 4xL4 pod (run_rw.sh), through an SSH tunnel, with E042's own sandbox container
# (arc-sandbox-e042). Same harness flags, sampler (temperature 1.0, top_p 1.0) and effort prefix as E043's
# e043_e128t_w10_u2_gptq_eng64_sub80 run (with --rescue-code: its jsonl records `rescues`, written only with the flag),
# so attempts pair up with it. Time-boxed by the caller; reruns resume.
# Usage: eval_rw.sh <tag> [extra harness args]   env: WORKERS (40), SANDBOX, BASE (http://127.0.0.1:18100/v1)
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
TAG=$1; shift
cd "$ROOT/evals"
EXTRA='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},"temperature":1.0,"top_p":1.0,"cache_salt":"'"$TAG"'"}'
exec ../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool \
  --final-mode session --lenient-parse --max-tokens 64000 --max-total-output 80000 --budget-soft 40000 \
  --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal --turns-note 5 \
  --train-check --rescue-code --attempts 2 --workers "${WORKERS:-40}" --sandbox-url "${SANDBOX:-http://172.18.0.6:6000}" \
  --base-url "${BASE:-http://127.0.0.1:18100/v1}" --model dsv41 --extra-body "$EXTRA" \
  --challenges "$ROOT/data/arc-agi-2-github/data/evaluation" --tag "e042_$TAG" "$@" \
  >> "results/log_e042_$TAG.txt" 2>&1
