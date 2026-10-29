#!/usr/bin/env bash
# E065: after the pilot, run all of nvarc_b0 (resumes the pilot's 150), then the first 3,570 nvarc_b1 (5K NVARC). Each launch stops
# starting attempts when the DeepSeek balance falls below $5 (re-run after a top-up resumes).
E=/home/vu/Workspace/ARC-AGI-2/experiments/E065_image_harness_rollouts
until grep -q "done nvarc_b0" $E/pilot.out 2>/dev/null; do sleep 30; done
$E/launch.sh nvarc_b0
$E/launch.sh nvarc_b1 $E/pairs_nvarc_b1_first3570.txt  # first 5K NVARC = b0 + these; the rest after review
