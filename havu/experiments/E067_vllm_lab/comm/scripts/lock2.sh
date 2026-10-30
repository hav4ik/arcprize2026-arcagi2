#!/bin/bash
# Runs "$@" holding both lab GPU locks (order gpu0 then gpu1, as the lab rule).
# Queues on gpu0, then waits at most 240 s for gpu1; on timeout it releases
# gpu0 for a while, so it never sits on an idle GPU for long.
L=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
while true; do
  flock "$L/gpu0.lock" flock -w 240 -E 75 "$L/gpu1.lock" "$@"
  rc=$?
  [ $rc -ne 75 ] && exit $rc
  sleep 20
done
