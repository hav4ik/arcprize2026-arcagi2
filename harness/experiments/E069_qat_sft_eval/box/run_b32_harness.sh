#!/bin/bash
# E068: B32's frozen tool harness (kaggle/dsv41_tools/pinned_candb: run_deepseek_tools.py 882c817f, train_check.py
# 8fe36175, auto_repair.py 2292369c + auto_repair_names.json 571f47eb) on top of the V5 kit's evals/ snapshot
# (kaggle_kit_v5/dsv41_kit.tgz repo/), i.e. exactly the module set B32's notebook runs. Tool code runs in the shared
# local Docker sandbox (not the Kaggle LocalExecutor). Flags = B32 solver HARNESS + make_args (multi-test, auto-repair).
# Needs the ssh tunnel: ssh -N -L 18068:127.0.0.1:18000 -p 61847 root@38.20.137.73
# Usage: run_b32_harness.sh <tag> [extra harness args, e.g. --attempts 1 --tasks a,b --workers 2]
#   env: BASE (http://127.0.0.1:18068/v1), MODEL (dsv41), SANDBOX (http://172.18.0.2:6000),
#        CHALLENGES (data/arc-agi-2-github/data/evaluation), SALT (= tag)
set -eu
TAG=$1; shift
ROOT=/home/vu/Workspace/ARC-AGI-2
HERE=$ROOT/experiments/E069_qat_sft_eval/box
K=$HERE/b32kit
if [ ! -f $K/repo/evals/auto_repair.py ]; then  # build the scratch kit once
  mkdir -p $K && tar xzf $ROOT/kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz -C $K repo
  cp $ROOT/kaggle/dsv41_tools/pinned_candb/{run_deepseek_tools.py,train_check.py,auto_repair.py,auto_repair_names.json} \
    $K/repo/evals/
fi
(cd $K/repo/evals && sha256sum run_deepseek_tools.py train_check.py auto_repair.py auto_repair_names.json | cut -c1-8 \
  | tr '\n' ' ' | grep -q '^882c817f 8fe36175 2292369c 571f47eb ') || { echo "pinned hashes differ"; exit 1; }
mkdir -p $HERE/results $HERE/logs
SALT=${SALT:-$TAG}
cd $K/repo/evals
exec $ROOT/.venv/bin/python run_deepseek_tools.py \
  --challenges ${CHALLENGES:-$ROOT/data/arc-agi-2-github/data/evaluation} \
  --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal \
  --turns-note 5 --train-check --rescue-code --token-exact --fallback-solve --multi-test --multi-budget 0.5 \
  --auto-repair --context-limit 163840 --max-turns 40 --model ${MODEL:-dsv41} --base-url ${BASE:-http://127.0.0.1:18068/v1} \
  --extra-body "{\"chat_template_kwargs\": {\"thinking\": true, \"reasoning_effort\": 50}, \"temperature\": 1.0, \"top_p\": 1.0, \"cache_salt\": \"$SALT\"}" \
  --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --tag $TAG --out $HERE/results/$TAG.jsonl "$@"
