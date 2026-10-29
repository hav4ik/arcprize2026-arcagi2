#!/usr/bin/env bash
# E065 text twin: arc2_train, all nvarc_b0, then the first 3,570 nvarc_b1 (the same 5,994 pairs as the image set).
E=/home/vu/Workspace/ARC-AGI-2/experiments/E065_image_harness_rollouts
$E/launch_text.sh arc2_train
$E/launch_text.sh nvarc_b0
$E/launch_text.sh nvarc_b1 $E/pairs_nvarc_b1_first3570.txt
