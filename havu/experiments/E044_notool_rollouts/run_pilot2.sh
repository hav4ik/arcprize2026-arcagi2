#!/usr/bin/env bash
# E044 pilot, resumed 2026-10-23 ~20:10 UTC with the DeepInfra/DeepSeek split (rest of nvarc_b0, then arc2_train retries).
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E044_notool_rollouts; E025=$ROOT/experiments/E025_train_rollouts/data
echo $$ > "$HERE/run.pid"
NO_PID=1 "$HERE/run_split.sh" nvarc_b0 "$E025/nvarc_b0_challenges.json" "$E025/nvarc_b0_solutions.json" "$HERE/pairs_nvarc_b0.txt"
NO_PID=1 "$HERE/run_split.sh" arc2_train "$ROOT/data/arc-agi_training_challenges.json" "$ROOT/data/arc-agi_training_solutions.json" "$HERE/pairs_arc2_train.txt"
echo "=== $(date -u +%FT%TZ) pilot done"
