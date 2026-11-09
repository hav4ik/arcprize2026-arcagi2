#!/usr/bin/env bash
# E044 continuation (2026-10-23 ~21:20 UTC): all 7,500 nvarc_b1 tasks split over DeepInfra, InferenceNet (OpenRouter)
# and the DeepSeek API, then a retry pass. The b0 retries finish separately.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E044_notool_rollouts; E025=$ROOT/experiments/E025_train_rollouts/data
echo $$ > "$HERE/run.pid"
export NO_PID=1 PROVIDERS=${PROVIDERS:-DeepInfra:128,InferenceNet:64,deepseek:48}
"$HERE/run_split.sh" nvarc_b1 "$E025/nvarc_b1_challenges.json" "$E025/nvarc_b1_solutions.json" "$HERE/pairs_nvarc_b1.txt"
"$HERE/run_split.sh" nvarc_b1 "$E025/nvarc_b1_challenges.json" "$E025/nvarc_b1_solutions.json" "$HERE/pairs_nvarc_b1.txt"
"$HERE/run_split.sh" nvarc_b0 "$E025/nvarc_b0_challenges.json" "$E025/nvarc_b0_solutions.json" "$HERE/pairs_nvarc_b0.txt"
echo "=== $(date -u +%FT%TZ) b1 done"
