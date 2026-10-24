#!/usr/bin/env bash
# E044 pilot: no-tool rollouts on the 2,500 E031 pairs (arc2_train, then nvarc_b0). Writes its pid for the uploader.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E044_notool_rollouts; E025=$ROOT/experiments/E025_train_rollouts/data
echo $$ > "$HERE/run.pid"
STOP=${STOP:-60} "$HERE/launch.sh" arc2_train "$ROOT/data/arc-agi_training_challenges.json" "$ROOT/data/arc-agi_training_solutions.json" "$HERE/pairs_arc2_train.txt"
STOP=${STOP:-60} "$HERE/launch.sh" nvarc_b0 "$E025/nvarc_b0_challenges.json" "$E025/nvarc_b0_solutions.json" "$HERE/pairs_nvarc_b0.txt"
