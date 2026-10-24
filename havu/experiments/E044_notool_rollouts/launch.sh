#!/usr/bin/env bash
# E044: no-tool Flash-1017 rollouts (E035 arm D: low + minimal header + task summary).
# PROVIDER=deepseek (DeepSeek API) or an OpenRouter provider name to pin (default DeepInfra; e.g. InferenceNet).
# Usage: launch.sh eval120 | launch.sh <name> <challenges> <solutions> <pairs_file>   (re-running resumes)
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2
cd "$ROOT/evals"
P=${PROVIDER:-DeepInfra}
COMMON="--effort low --effort-header minimal --task-summary --workers ${WORKERS:-64} --max-balance-stop ${STOP:-20}"
if [ "$P" = deepseek ]; then
  CFG="--model deepseek-flash $COMMON"
else
  CFG="--model deepseek/deepseek-v4.1-flash-1017 --openrouter-provider $P --context 131072 $COMMON"
fi
p=$(echo "$P" | tr 'A-Z' 'a-z')
if [ "$1" = eval120 ]; then
  echo "=== $(date -u +%FT%TZ) start eval120 ($P)"
  ../.venv/bin/python run_deepseek_arcprize.py $CFG --attempts 2 --out results/e044_notool_${p}_eval120.jsonl \
    >> results/log_e044_eval120.txt 2>&1
else
  if [ ! -s "$4" ]; then echo "=== $(date -u +%FT%TZ) skip $1 ($P): empty pair list"; exit 0; fi
  echo "=== $(date -u +%FT%TZ) start $1 ($P)"
  ../.venv/bin/python run_deepseek_arcprize.py $CFG --attempts 1 --challenges "$2" --solutions "$3" --pairs "$(cat "$4")" \
    --out "results/e044_notool_${p}_$1.jsonl" >> "results/log_e044_${p}_$1.txt" 2>&1
fi
echo "=== $(date -u +%FT%TZ) done $1 ($P)"
