#!/bin/bash
# every $1 s: poll all pods, append to the timeline, print rows; print new setup-log lines
D=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad
P=$D/podwatch; INT=${1:-600}
seen=$(cat $P/setup_*.log 2>/dev/null | wc -l)
while true; do
  for i in $(seq 1 $((INT / 30))); do
    n=$(cat $P/setup_*.log 2>/dev/null | wc -l)
    if [ "$n" -gt "$seen" ]; then cat $P/setup_*.log | tail -n $((n - seen)) | sed 's/^/SETUP /'; seen=$n; fi
    sleep 30
  done
  $P/poll.sh > $P/last_poll.txt 2>&1
  /home/vu/Workspace/ARC-AGI-2/.venv/bin/python $P/parse.py $D/pods_timeline.md $P/timeline.tsv < $P/last_poll.txt 2>&1 | sed 's/^/ROW /'
done
