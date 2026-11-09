#!/usr/bin/env bash
# E032 eval: one arm (served model name + server URLs) through the DeepSeek harness with E031's exact config
# (s20x60 + --effort-header minimal + --turns-note 5, lean2sub, answer tool, train-check), Qwen sampling, xhigh.
# Pair sets in order: dev56, then the 93 E027 Kaggle-pace pairs, then the remaining 23. Re-running resumes.
#   eval_chain.sh MODEL URLS [WORKERS] [SETS]   (SETS default "dev56 k93 rest"; WAIT_PID=pid waits for it first;
#   ATTEMPTS=2 resumes the same files and adds attempt #1 for pass@2; EFFORT=medium|low|minimal sets the Qwen tier
#   (minimal needs chat_template_minimal.jinja on the server) and TAGX a tag suffix, e.g. _effmin)
set -uo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
HERE=$ROOT/experiments/E032_qwen38_sft_e031
MODEL=$1; URLS=$2; W=${3:-20}; SETS=${4:-dev56 k93 rest}
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done; fi
cd "$ROOT/evals"
CFG="--mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong \
  --effort-header minimal --turns-note 5 --train-check --attempts ${ATTEMPTS:-1} --sandbox-url http://172.18.0.2:6000 \
  --challenges ../data/arc-agi_evaluation_challenges.json --solutions ../data/arc-agi_evaluation_solutions.json \
  --allow-expensive-run --workers $W"
EB='{"temperature":1.0,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"reasoning_effort":"'${EFFORT:-xhigh}'"}}'
for set in $SETS; do
  echo "=== $(date -u +%FT%TZ) start $MODEL $set"
  ../.venv/bin/python run_deepseek_tools.py --base-url "$URLS" --model "$MODEL" --extra-body "$EB" $CFG \
    --pairs "$(cat "$HERE/pairs_$set.txt")" --tag "e032_$set${TAGX:-}" >> "results/log_e032_${MODEL}_$set${TAGX:-}.txt" 2>&1
  echo "=== $(date -u +%FT%TZ) done $MODEL $set rc=$?"
done
