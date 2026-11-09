#!/usr/bin/env bash
# E035: no-tool shortening arms (ARC Prize prompt, 120 tasks / 167 pairs x 2 attempts), launched in parallel.
# Waits until the weekday peak window ends (10:05 UTC) if started inside it.
set -euo pipefail
cd /home/vu/Workspace/ARC-AGI-2/evals
now=$(date -u +%s); start=$(date -u -d 'today 10:05' +%s)
h=$(date -u +%H); dow=$(date -u +%u)
if [ "$dow" -le 5 ] && [ "$h" -ge 6 ] && [ "$now" -lt "$start" ]; then sleep $((start - now)); fi
P="../.venv/bin/python run_deepseek_arcprize.py --model deepseek-flash --attempts 2 --workers 40"
$P --effort max --task-summary --tag e035_sum > results/log_arcprize_e035_max_sum.txt 2>&1 &
$P --effort low --tag e035_plain > results/log_arcprize_e035_low_plain.txt 2>&1 &
$P --effort low --task-summary --tag e035_sum > results/log_arcprize_e035_low_sum.txt 2>&1 &
$P --effort low --task-summary --effort-header minimal --tag e035_hdrsum > results/log_arcprize_e035_low_hdrsum.txt 2>&1 &
$P --effort low --effort-header minimal --tag e035_hdr > results/log_arcprize_e035_low_hdr.txt 2>&1 &
echo "launched $(date -u +%FT%TZ)"; wait; echo "all done $(date -u +%FT%TZ)"
