#!/usr/bin/env bash
# E043: ARC-AGI-2 eval (120 tasks / 167 pairs, 1 attempt) of the self-hosted model on the 8x box, with the
# E031/E036 calibration harness config (s20x60 + minimal effort header, persistent-v6), run locally against
# an SSH tunnel to the box's vLLM server and the local arc-sandbox-e043 container.
# Usage: eval_arc.sh <tag> [extra harness args, e.g. --limit 40 or --attempts 2]
#   env: WORKERS (48), SANDBOX (http://172.18.0.4:6000), BASE (http://127.0.0.1:18000/v1),
#        PRESET: s20x60 (default; 60k total, 20k soft, strong wording) or sub80s40 (80k total, 40k soft; E030
#        "sub80s40 + header", the best-accuracy API config: pass@2 84.9% vs 80.3%). Both use the minimal effort header.
# Effort: integer 50 (= encoding.py's "low"), the same prefix the calibration rollouts were tokenized with.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
TAG=$1; shift
cd "$ROOT/evals"
case "${PRESET:-s20x60}" in
  s20x60) BUDGET="--max-total-output 60000 --budget-soft 20000 --soft-wording strong" ;;
  sub80s40) BUDGET="--max-total-output 80000 --budget-soft 40000" ;;
  *) echo "unknown PRESET ${PRESET}" >&2; exit 2 ;;
esac
# SALT: per-run cache_salt so a runtime-mask server never reuses prefixes cached under another mask.
EXTRA='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},"temperature":1.0,"top_p":1.0,"cache_salt":"'"${SALT:-$TAG}"'"}'
../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool \
  --final-mode session --lenient-parse --max-tokens 64000 $BUDGET --budget-notes 0.01 \
  --budget-warn 0.75 --effort-header minimal --turns-note 5 \
  --train-check --attempts 1 --workers "${WORKERS:-48}" --sandbox-url "${SANDBOX:-http://172.18.0.4:6000}" \
  --base-url "${BASE:-http://127.0.0.1:18000/v1}" --model dsv41 --extra-body "$EXTRA" \
  --challenges "$ROOT/data/arc-agi-2-github/data/evaluation" --tag "e043_$TAG" "$@" \
  >> "results/log_e043_$TAG.txt" 2>&1
