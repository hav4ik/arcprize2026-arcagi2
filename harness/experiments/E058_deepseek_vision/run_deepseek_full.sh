#!/usr/bin/env bash
# E058 DeepSeek image arm on all 120 eval tasks: resumes e058_pilot30 (only missing tasks run), off-peak only.
# --allow-expensive-run: the guard checks the worst case (all attempts at the 80k cap at peak price, $17); the
# expected cost from the pilot is ~$3.3 off-peak, and the $15 balance kill-guard stays on.
X=/home/vu/Workspace/ARC-AGI-2/experiments/E058_deepseek_vision
h=$(date -u +%H); d=$(date -u +%u)
if [ "$d" -le 5 ] && { [ "$h" -ge 1 -a "$h" -lt 4 ] || [ "$h" -ge 6 -a "$h" -lt 10 ]; }; then echo "peak window, not launching"; exit 1; fi
$X/run_pilot.sh e058_pilot30 --attempts 2 --allow-expensive-run
