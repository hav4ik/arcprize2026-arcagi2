#!/usr/bin/env bash
# E066: ARC-AGI-2 eval (120 tasks, 2 attempts) of a self-hosted model in one of the two calibration harnesses:
#   HARNESS=img (default): the E058 image harness of the E065 image rollouts: E046 best (sub80s40 + minimal effort
#     header + turns note 5 + fallback solve + multi-test 0.5) + one token-aligned image per grid; chat path
#     (--token-exact has no image path). The server must keep the vision tower (VISION=1).
#   HARNESS=txt: the text twin = E046 best as on Kaggle, with --token-exact (AGENTS.md: every self-hosted
#     DeepSeek-V4.1 tool run) and --context-limit.
# Run locally against an SSH tunnel to the box's vLLM server and a local sandbox container.
# Usage: eval_arc_img.sh <tag> [extra harness args, e.g. --limit 20]
#   env: HARNESS (img), WORKERS (120), SANDBOX (http://172.18.0.4:6000), BASE (http://127.0.0.1:18000/v1),
#        ATTEMPTS (2), SAMPLING (default "temperature":1.0,"top_p":1.0), SALT (default: the tag; one per mask config),
#        MAXLEN (131072, the server's --max-model-len, for --context-limit)
# Effort: integer 50 (= encoding.py's "low"), the prefix the calibration rollouts were tokenized with.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
TAG=$1; shift
cd "$ROOT/evals"
case "${HARNESS:-img}" in
  img) GRID="--grid-format text+image --image-cell 0" ;;
  txt) GRID="--token-exact --context-limit ${MAXLEN:-131072}" ;;
  *) echo "unknown HARNESS ${HARNESS}" >&2; exit 2 ;;
esac
EXTRA='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},'"${SAMPLING:-\"temperature\":1.0,\"top_p\":1.0}"',"cache_salt":"'"${SALT:-$TAG}"'"}'
../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool \
  --final-mode session --lenient-parse --max-tokens 64000 --max-total-output 80000 --budget-soft 40000 \
  --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal --max-turns 40 --turns-note 5 --fallback-solve \
  --multi-test --multi-budget 0.5 --train-check $GRID --rescue-code \
  --attempts "${ATTEMPTS:-2}" --workers "${WORKERS:-120}" --sandbox-url "${SANDBOX:-http://172.18.0.4:6000}" \
  --base-url "${BASE:-http://127.0.0.1:18000/v1}" --model dsv41 --extra-body "$EXTRA" \
  --challenges "$ROOT/data/arc-agi-2-github/data/evaluation" --tag "e066_$TAG" "$@" \
  >> "results/log_e066_$TAG.txt" 2>&1
