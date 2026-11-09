#!/usr/bin/env bash
# E043 no-tool round: ARC-AGI-2 eval (120 tasks / 167 pairs, 1 attempt) of the self-hosted model with the E044
# no-tool config (E035 arm D: effort low + minimal header + task summary, official ARC Prize prompt, single turn),
# run locally against an SSH tunnel to the box's vLLM server. The mask is whatever the server's ctrl file holds;
# set it first (runtime masks) or serve an export.
# Usage: eval_notool.sh <tag> [extra harness args, e.g. --limit 40]
#   env: WORKERS (167), BASE (http://127.0.0.1:18000/v1), SALT (tag)
# Effort: integer 50 (= encoding.py's "low"), the prefix the no-tool calibration rollouts were tokenized with.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
TAG=$1; shift
cd "$ROOT/evals"
# SALT: per-run cache_salt so a runtime-mask server never reuses prefixes cached under another mask.
EXTRA='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},"temperature":1.0,"top_p":1.0,"cache_salt":"'"${SALT:-$TAG}"'"}'
../.venv/bin/python run_deepseek_arcprize.py --base-url "${BASE:-http://127.0.0.1:18000/v1}" --model dsv41 \
  --effort low --effort-header minimal --task-summary --attempts 1 --workers "${WORKERS:-167}" --context 131072 \
  --extra-body "$EXTRA" --out "results/e043nt_$TAG.jsonl" "$@" >> "results/log_e043nt_$TAG.txt" 2>&1
