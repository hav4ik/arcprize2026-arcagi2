#!/bin/bash
# poll now, then every $1 s
P=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/podwatch; INT=${1:-600}
while true; do
  $P/poll.sh > $P/last_poll.txt 2>&1
  /home/vu/Workspace/ARC-AGI-2/.venv/bin/python $P/parse.py $P/../pods_timeline.md $P/timeline.tsv < $P/last_poll.txt 2>&1 | sed 's/^/ROW /'
  echo POLLDONE; sleep $INT
done
