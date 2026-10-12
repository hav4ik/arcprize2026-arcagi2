#!/usr/bin/env bash
# Usage: run_variant.sh <tag> <workers> <pairs-file|full> <attempts> [extra harness args...]
# Runs evals/run_deepseek_tools.py in grid mode at low effort, off-peak pricing, logging to evals/results/log_<tag>.txt.
set -euo pipefail
TAG=$1; WORKERS=$2; SET=$3; ATTEMPTS=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEL=()
if [ "$SET" != full ]; then SEL=(--pairs "$(cat "$SET")"); fi
cd "$ROOT/evals"
exec ../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --price-multiplier 1 \
  --attempts "$ATTEMPTS" --workers "$WORKERS" --tag "$TAG" "${SEL[@]}" "$@" \
  >> "$ROOT/evals/results/log_$TAG.txt" 2>&1
