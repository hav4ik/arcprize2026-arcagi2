#!/bin/bash
# E069 API row: B32's frozen harness (same kit as run_b32_harness.sh, pinned hashes checked) on the FULL DeepSeek-V4.1
# Flash-1017 via OpenRouter -> DeepInfra (fp8; parity with the DeepSeek API in E037/E038). Same flags as B32 except the
# three that only apply to a self-hosted server: --token-exact, --context-limit, --base-url/--extra-body (sampling is
# the provider's default; --effort low = DeepSeek's numeric effort 50, as in B32).
set -eu
TAG=$1; shift
ROOT=/home/vu/Workspace/ARC-AGI-2
HERE=$ROOT/experiments/E069_qat_sft_eval/box
K=$ROOT/data/e069_b32kit  # outside the synced tree: its repo/SECRETS.yaml is a symlink to the root file (the pinned harness reads ROOT/SECRETS.yaml)
(cd $K/repo/evals && sha256sum run_deepseek_tools.py train_check.py auto_repair.py auto_repair_names.json | cut -c1-8 \
  | tr '\n' ' ' | grep -q '^882c817f 8fe36175 2292369c 571f47eb ') || { echo "pinned hashes differ"; exit 1; }
cd $K/repo/evals
exec $ROOT/.venv/bin/python run_deepseek_tools.py \
  --challenges $ROOT/data/arc-agi_evaluation_challenges.json \
  --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000 \
  --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal \
  --turns-note 5 --train-check --rescue-code --fallback-solve --multi-test --multi-budget 0.5 \
  --auto-repair --max-turns 40 --model deepseek/deepseek-v4.1-flash-1017 --openrouter-provider DeepInfra \
  --sandbox-url ${SANDBOX:-http://172.18.0.2:6000} --max-balance-stop ${STOP:-60} --allow-expensive-run \
  --tag $TAG --out $HERE/results/$TAG.jsonl "$@"
