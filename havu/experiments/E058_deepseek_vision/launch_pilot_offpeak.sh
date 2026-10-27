#!/usr/bin/env bash
# E058: wait for DeepSeek off-peak (10:00 UTC on weekdays), then run the 30-task pilot (2 attempts) with images.
X=/home/vu/Workspace/ARC-AGI-2/experiments/E058_deepseek_vision
until [ "$(date -u +%H)" -ge 10 ]; do sleep 60; done
echo "launch $(date -u +%H:%M)"
$X/run_pilot.sh e058_pilot30 --tasks "$(cat $X/pilot_tasks.txt)" --attempts 2
echo "done $(date -u +%H:%M)"
